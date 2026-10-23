# Run with: python test_passage_prep.py   (plain asserts, no pytest)
from passage_prep import (clean_passage_text, split_into_sentences, build_premises, validate_example,
                          build_pairs, coverage_ratio)


class _Ex:
    def __init__(self, passages):
        self.passages = passages


def test_markers_removed():
    out = clean_passage_text("<s>Hello world.</s><s>Second one.</s>")
    assert "<s>" not in out and "</s>" not in out
    assert out == "Hello world. Second one."


def test_unclosed_cite_template_removed():
    # the unclosed opener, its fields and the URL go, but the text after it is kept (not swallowed to the end)
    out = clean_passage_text("<s>He wrote a book. {{cite web|url=http://foo.org/x|title=Bar and more text that runs on")
    assert out.startswith("He wrote a book."), out
    assert "{{" not in out and "cite" not in out and "http" not in out and "url=" not in out and "title=" not in out
    assert "more text that runs on" in out


def test_unclosed_template_keeps_following_prose_jahanbani():
    raw = ("<s>Shahnaz Pahlavi married Ardeshir Zahedi on 14 October 1957.{{cite news|title=Iran Shah's daughter to wed "
           "engineer in simple ceremony|url=https://news.google.com/newspapers?id=Zc&sjid=8v&pg=1761,3998465&dq=m&hl=en"
           "|access-date=23 July 2013|newspaper=Lewiston Evening Journal Shahnaz later married Khosrow Jahanbani in "
           "February 1971 at the Iranian Embassy, Paris.{{cite web|title=Centers of Power in Iran During her father's "
           "reign, Shahnaz had investments in agricultural enterprises.</s>")
    out = clean_passage_text(raw)
    assert "Khosrow Jahanbani" in out, out
    assert "Iranian Embassy, Paris." in out and "agricultural enterprises." in out
    for junk in ("{{", "http", "url=", "newspaper=", "access-date"):
        assert junk not in out, (junk, out)


def test_unclosed_template_keeps_following_prose_belgian_school():
    raw = ("<s>Early life and education. Shahnaz Pahlavi was born in Tehran on 27 October 1940.{{cite news|title=Persian "
           "title|url=https://www.bbc.com/persian/iran/2013/12/131210_u08_egypt_shahnaz|access-date=7 August 2021|work=BBC"
           "|date=11 December 2013 Shahnaz Pahlavi was educated in a Belgian boarding school, the Lyc\u00e9e, in Li\u00e8ge "
           "and then in Switzerland.</s><s>Personal life. Her father had plans.</s>")
    out = clean_passage_text(raw)
    assert "educated in a Belgian boarding school" in out, out
    assert "Her father had plans." in out
    assert "http" not in out and "date=" not in out and "work=" not in out


def test_closed_templates_still_fully_removed():
    assert clean_passage_text("A {{cite web|title=T. Next Sentence|url=http://x.y}} B.") == "A B."
    assert clean_passage_text("A {{outer|x={{inner|y. Z}}|z. Q}} B") == "A B"
    assert clean_passage_text("Before {{a}} middle {{b|c}} after.") == "Before middle after."
    # a closed template followed by an unclosed one: only the unclosed opener is stripped
    out = clean_passage_text("One. {{closed|x}} Two. {{cite news|title=Foo and then real prose.")
    assert out.startswith("One. Two.") and "real prose." in out and "closed" not in out


def test_removed_span_flags():
    r = validate_example(_Ex([{"text": "<s>Hello there. {{cite|title=A. Big Name|x=y}} Bye now.</s>"}]))
    assert r["swallowed_sentences"] is True and r["removed_span_words"] == 3, r
    r = validate_example(_Ex([{"text": "<s>Hello there. {{cite|title=a b c}} Bye now.</s>"}]))
    assert r["swallowed_sentences"] is False and r["removed_span_words"] == 3, r
    r = validate_example(_Ex([{"text": "<s>Plain text only.</s>"}]))
    assert r["swallowed_sentences"] is False and r["removed_span_words"] == 0
    # an unclosed template removes only its opener, so nothing is "swallowed"
    r = validate_example(_Ex([{"text": "Start. {{cite news|title=Foo bar baz. Real prose here."}]))
    assert r["swallowed_sentences"] is False and r["removed_span_words"] <= 2, r


def test_closed_and_nested_template_removed():
    assert clean_passage_text("A {{outer|x={{inner|y}}|z}} B") == "A B"


def test_url_removed():
    out = clean_passage_text("See https://example.com/a?b=1 and www.foo.org/x for details.")
    assert "http" not in out and "www" not in out
    assert out == "See and for details.", out


def test_repeated_title_dropped():
    assert clean_passage_text("Lanny Flaherty Lanny Flaherty (born 1942) is an actor.") == \
        "Lanny Flaherty (born 1942) is an actor."
    assert clean_passage_text("Marianne McAndrew Marianne Christine McAndrew is an actress.") == \
        "Marianne Christine McAndrew is an actress."
    assert clean_passage_text("John Smith is a man.") == "John Smith is a man."


def test_abbreviations_not_split():
    assert split_into_sentences("Mr. Smith met Dr. Jones.") == ["Mr. Smith met Dr. Jones."]
    assert split_into_sentences("J. K. Rowling wrote it.") == ["J. K. Rowling wrote it."]
    assert split_into_sentences("He was born c. 1950 in Ohio.") == ["He was born c. 1950 in Ohio."]
    assert split_into_sentences("It cost 3.5 dollars. Then it rose.") == ["It cost 3.5 dollars.", "Then it rose."]
    assert split_into_sentences("He left. She stayed.") == ["He left.", "She stayed."]


def test_heading_split_off():
    assert split_into_sentences("Early life. He was born in Ohio.") == ["He was born in Ohio."]
    assert split_into_sentences("Club career.: He joined Ajax.") == ["He joined Ajax."]


def test_empty_and_none_passages():
    assert build_premises([{"text": ""}]) == []
    assert build_premises([{"text": None}]) == []
    assert build_premises([{}]) == []
    assert build_premises([]) == []
    assert split_into_sentences("") == [] and split_into_sentences(None) == []
    assert clean_passage_text(None) == "" and clean_passage_text("") == ""


def test_duplicate_premises_removed():
    p = {"text": "<s>He ran. He ran.</s>"}
    assert build_premises([p, p]) == [("He ran. He ran.", 0)]
    out = build_premises([{"text": "Alpha is here. Beta is there."}, {"text": "alpha is here.  Beta is there."}])
    assert len(out) == 1 and out[0][1] == 0


def test_no_markers_still_splits():
    out = build_premises([{"text": "Alice went home. Bob stayed out. Carol slept in."}], window=1)
    assert [p for p, _ in out] == ["Alice went home.", "Bob stayed out.", "Carol slept in."]
    out2 = build_premises([{"text": "Alice went home. Bob stayed out. Carol slept in."}], window=2)
    assert [p for p, _ in out2] == ["Alice went home. Bob stayed out.", "Bob stayed out. Carol slept in."]


def test_max_words_and_index():
    long = " ".join(["word"] * 400) + "."
    out = build_premises([{"text": "Short one. " + long}, {"text": "Another here."}], max_words=50)
    assert all(len(p.split()) <= 50 for p, _ in out)
    assert out[-1] == ("Another here.", 1)


def test_validate_example():
    r = validate_example(_Ex([{"text": "<s>Foo Foo is a man. He ran.</s>"}, {"text": ""}]))
    assert r["n_passages"] == 2 and r["n_empty_passages"] == 1 and r["n_premises"] >= 1
    assert not r["empty_after_cleaning"] and not r["leftover_markup"] and not r["tokens_lost"]
    r = validate_example(_Ex([]))
    assert r["empty_after_cleaning"] and r["n_premises"] == 0
    assert isinstance(validate_example(None), dict)
    assert isinstance(validate_example(_Ex([{"text": 5}])), dict)


def test_build_pairs():
    ps = [{"text": "Alice went home. Bob stayed out. Carol slept in."}, {"text": ""}, {"text": "Dan ran far."}]
    pairs = build_pairs("  Alice   went\nhome.  ", ps)
    assert [(p["premise"], p["passage_index"]) for p in pairs] == build_premises(ps)
    assert all(p["hypothesis"] == "Alice went home." for p in pairs)
    assert set(pairs[0]) == {"premise", "hypothesis", "passage_index"}
    assert pairs[-1]["passage_index"] == 2
    assert build_pairs("x", []) == [] and build_pairs("x", [{"text": None}]) == []
    assert [p["premise"] for p in build_pairs("x", ps[:1], window=1)] == \
        ["Alice went home.", "Bob stayed out.", "Carol slept in."]


def test_coverage_ratio():
    # markers, closed + unclosed templates, URLs are excluded from the denominator -> full coverage
    assert coverage_ratio([{"text": "<s>He ran.</s> {{cite|url=http://x.y|title=T}} See http://a.b now {{open|never closed"}]) == 1.0
    # dropping a repeated title is a genuine loss: 2 of 8 words ("Lanny Flaherty") are lost
    r = coverage_ratio([{"text": "Lanny Flaherty Lanny Flaherty is an American actor."}])
    assert abs(r - 6 / 8) < 1e-9, r
    assert coverage_ratio([]) == 1.0 and coverage_ratio([{"text": ""}, {"text": None}, {}]) == 1.0
    assert coverage_ratio([{"text": "12345 {{only|template}}"}]) == 1.0
    assert 0.0 <= coverage_ratio([{"text": "Marianne McAndrew Marianne Christine McAndrew is an actress."}]) <= 1.0
    assert "coverage" in validate_example(_Ex([{"text": "<s>Foo is a man.</s>"}]))
    assert validate_example(_Ex([{"text": "<s>Foo is a man.</s>"}]))["coverage"] == 1.0


def test_prose_wrapped_in_closed_template_is_kept():
    raw = ("<s>Highway brought a lot of appreciation from the industry.{{cite web Nadiadwala made his directorial debut "
           "with the July 2014 release \"Kick\", a remake of Ravi Teja's 2009 Telugu film \"Kick\". The film stars Salman "
           "Khan and Jacqueline Fernandez as the leads. The film broke all previous Khan records and is one of the top "
           "100 highest-grossing Indian films. }}{{cite web }} Nadiadwala with this film also resurrected Fernandez's "
           "career.</s>")
    out = clean_passage_text(raw)
    assert "The film stars Salman Khan" in out, out
    assert "Nadiadwala made his directorial debut" in out and "Nadiadwala with this film" in out
    assert "{{" not in out and "}}" not in out and "cite" not in out.lower(), out
    r = validate_example(_Ex([{"text": raw}]))
    assert r["swallowed_sentences"] is False and not r["leftover_markup"], r


def test_structured_and_short_closed_templates_still_removed():
    assert clean_passage_text("A {{cite web|title=A. Big Name|url=http://x.y}} B.") == "A B."
    assert clean_passage_text("A {{cite web title is A. Big Name but no pipes at all in here ok}} B.") != "A B."  # >8 words, prose
    assert clean_passage_text("A {{cite web }} B.") == "A B."
    assert clean_passage_text("A {{x=y and more words than eight here in this long template ok yes}} B.") == "A B."
    long_structured = "A {{cite web|" + "word " * 30 + "}} B."
    assert clean_passage_text(long_structured) == "A B."


def test_leading_url_tail_removed():
    raw = ("07/03/exclusive-matthew-tukakis-job-history-not-checked-before-govt-role/|title=Exclusive: Matthew Tukaki's "
           "job history not checked before Govt role Tukaki has been active in his support for business.")
    out = clean_passage_text(raw)
    assert "07/03/exclusive" not in out and "role/" not in out, out
    assert out.startswith("Exclusive: Matthew Tukaki's job history"), out
    assert clean_passage_text("<s>/2020/07/some-page/ Then real text here.</s>") == "Then real text here."
    # ordinary slashes are untouched
    assert clean_passage_text("AC/DC is a band. and/or so.") == "AC/DC is a band. and/or so."
    assert clean_passage_text("He lived at 12/14 Main Street.") == "He lived at 12/14 Main Street."


def test_replacement_character_words_dropped():
    out = clean_passage_text("\ufffdlvarez stated he would fight. Later \u00c1lvarez won the bout. Mid\ufffdword stays out.")
    assert "\ufffd" not in out
    assert "lvarez stated" not in out and "Mid" not in out
    assert "Later \u00c1lvarez won the bout." in out, out
    assert "he would fight." in out
    prem = [p for p, _ in build_premises([{"text": "It began. \ufffdlvarez stated this. \u00c1lvarez won."}])]
    assert prem and all("\ufffd" not in p for p in prem)
    r = validate_example(_Ex([{"text": "It began. \ufffdlvarez stated this. \u00c1lvarez won."}]))
    assert not r["tokens_lost"] and not r["leftover_markup"], r


def test_leaked_pipe_key_has_no_stray_pipe():
    out = clean_passage_text("the squad |publisher=Liverpool F.C. On 8 September, he scored.")
    assert out == "the squad Liverpool F.C. On 8 September, he scored.", out
    assert not validate_example(_Ex([{"text": "the squad |publisher=Liverpool F.C. On 8 September, he scored."}]))["tokens_lost"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
