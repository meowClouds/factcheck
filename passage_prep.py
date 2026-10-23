# passage_prep.py
#
# Helpers that turn raw retrieved Wikipedia chunks into clean, sentence-window premises, e.g. for entailment.
# Standard library only (no NLTK punkt / spaCy models).

import re
from typing import List, Tuple

_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s|{}]+", re.IGNORECASE)
# residue of template parameters that leaked out of a {{...}}, e.g. "url=foo bar|" or "| title=Foo |"
# (the value may not cross a sentence boundary, so a missing closing "|" cannot swallow real prose)
_KEYVAL_RE = re.compile(r"\|?\s*\b[A-Za-z_][\w-]{0,30}\s*=\s*(?:(?!\.\s+[A-Z])[^|=]){0,200}?\|")
# a bare "key=" token left over once its value has no closing pipe, e.g. "date=11 December 2013 Shahnaz ..."
_BARE_KEY_RE = re.compile(r"\|?\b[A-Za-z_][\w-]{0,30}=(?=\S)")
# "{{cite news|" -- the opener of a template together with its name
_OPENER_RE = re.compile(r"\{\{[^|{}]{0,30}\|")
_SENTENCE_BOUNDARY_RE = re.compile(r"\.\s+[A-Z]")
# name of a prose-bearing template, e.g. "cite web" in "{{cite web Nadiadwala made his debut ...}}"
_CITE_NAME_RE = re.compile(r"\s*[Cc]ite(?:[ \t]+[a-z]\w*)?")
# leading fragment of a URL cut by the chunker: "07/03/exclusive-...-role/|" (whitespace-free, has "/", ends in | or /)
_URL_TAIL_RE = re.compile(r"^(?:[^\s|]*/[^\s|]*\||[^\s|]*/(?=\s|$))")
_MAX_STRUCTURELESS_WORDS = 8
_STRAY_PIPES_RE = re.compile(r"\s\|+\s|\|{2,}")
_WS_RE = re.compile(r"\s+")
_MAX_TITLE_WORDS = 6

_PLACEHOLDER = "\x00"

_TITLE_ABBREVS = (
    "Mr|Mrs|Ms|Mx|Dr|St|Jr|Sr|Prof|Mt|Gen|Col|Lt|Sgt|Capt|Cpt|Maj|Cmdr|Adm|Rev|Hon|Gov|Sen|Rep|Pres|Fr|Br|Sr|Messrs|Mme|Mlle"
)
# Protected whenever they appear, even before a capital letter.
_ALWAYS_PROTECT = re.compile(r"\b(?:%s)\." % _TITLE_ABBREVS)
# Protected only when followed by a number: "c. 1950", "No. 5", "Jan. 5", "p. 12".
_BEFORE_NUMBER = re.compile(
    r"\b(?:c|ca|b|d|fl|cca|No|Nos|p|pp|vol|ch|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s*\d)"
)
# Protected only when the next word is lowercase / punctuation / digit: "Acme Inc. is", "vs. the", "e.g. a".
_BEFORE_LOWER = re.compile(r"\b(?:Inc|Ltd|Co|Corp|vs|v|approx|est|al|etc)\.(?=\s*[a-z0-9,;)])")
# Single-letter initials: "J. K. Rowling", "John F. Kennedy".
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s*[A-Z]\b|\s*[A-Z][a-z])")
# Dotted acronyms (U.S., a.m., Ph.D.) are protected unless a capitalised word follows the last dot.
_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}(?!\s+[A-Z])")
_INLINE_DOTTED = re.compile(r"\b(?:[A-Za-z]\.){2,}(?=[A-Za-z])")

# A sentence boundary: terminal punctuation (+ optional closers / the ".:" of "Club career.:") then whitespace
# and something that can start a sentence.
_BOUNDARY_RE = re.compile(r"""([.!?]+[:"'”’)\]]*)(?:\s+|(?<=:))(?=["'“‘(\[]?[A-Z0-9])""")

_HEADING_PHRASES = {
    "early life", "early life and education", "early life and career", "life and career", "personal life",
    "later life", "late life", "career", "club career", "international career", "senior career", "youth career",
    "playing career", "professional career", "political career", "military career", "acting career",
    "early career", "later career", "career statistics", "biography", "background", "education", "family",
    "death", "legacy", "honours", "honors", "awards", "awards and honours", "awards and honors", "history",
    "filmography", "discography", "works", "publications", "bibliography", "politics", "personal", "overview",
    "life", "life and work", "early years", "music career", "film career", "television career", "career highlights",
    "playing style", "retirement", "coaching career", "managerial career", "recognition", "see also", "references",
    "notes", "references and notes", "external links", "further reading", "trivia", "controversy", "controversies",
    "television", "film", "films", "theatre", "theater", "radio", "stage", "video games", "criticism", "reception", "influence", "works and publications", "selected works", "selected publications",
}
_HEADING_LAST_WORDS = {"life", "career", "career.", "years", "background", "biography", "education", "history",
                       "honours", "honors", "awards", "legacy", "filmography", "discography", "bibliography"}


def clean_passage_text(text: str) -> str:
    """
    Cleans one retrieved passage chunk.

    - removes the <s> / </s> sentence markers
    - removes structured closed {{...}} wiki templates completely (nesting handled; see _strip_templates_with_spans
      for the exception of long prose-bearing ones); an unclosed "{{" is NOT swallowed to the
      end of the chunk: only its "{{name|" opener, URLs and "key=" tokens go, and the following prose is kept
      (some leftover title / field words are accepted as noise)
    - removes URLs and leaked template residue such as "url=foo|" / "title=Bar |" and stray "|"
    - removes a leading URL fragment cut by the chunker (a whitespace-free token with "/" ending in "|" or "/")
    - drops (does not repair) whitespace-delimited words containing the replacement character U+FFFD
    - collapses all whitespace runs to single spaces
    - drops a repeated leading title, so "Lanny Flaherty Lanny Flaherty (born ...)" becomes
      "Lanny Flaherty (born ...)" and "Marianne McAndrew Marianne Christine McAndrew is ..." becomes
      "Marianne Christine McAndrew is ..."

    :param text: raw "text" field of a passage dict
    :return: the cleaned text (possibly empty)
    """
    if not text:
        return ""
    return _drop_repeated_title(_remove_markup(text))


def _remove_markup(text: str) -> str:
    """
    Removes markers, templates, URLs, key=value| residue, a leading URL fragment and words containing the Unicode
    replacement character, and collapses whitespace (the repeated title is not dropped here).
    """
    text = text.replace("<s>", " ").replace("</s>", " ")
    text = _strip_templates(text)
    text = _URL_RE.sub(" ", text)
    text = _URL_TAIL_RE.sub(" ", text.lstrip()).lstrip()
    text = _KEYVAL_RE.sub(" ", text)
    text = _BARE_KEY_RE.sub("", text)
    text = _STRAY_PIPES_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if "\ufffd" in text:  # words damaged by a broken character are dropped, not repaired
        text = " ".join(w for w in text.split(" ") if "\ufffd" not in w)
    return text


def _drop_repeated_title(text: str) -> str:
    """
    Chunks start with the page title followed by the article's own opening ("Marianne McAndrew Marianne Christine
    McAndrew is ..."). If the first k words (k <= 6, capitalised) reappear right after themselves -- same first word,
    same last word within the next k+2 words -- the leading copy is dropped.
    """
    words = text.split(" ")
    for k in range(1, min(_MAX_TITLE_WORDS, len(words) // 2) + 1):
        head, rest = words[:k], words[k:]
        if not all(w[:1].isupper() for w in head):
            break
        if rest[0] == head[0] and head[-1] in rest[:k + 2] and (k > 1 or rest[0] == head[0] and len(rest) > 1 and rest[1] == head[0]):
            return " ".join(rest)
    return text


def _strip_templates(text: str) -> str:
    """Removes {{...}} templates; see _strip_templates_with_spans."""
    return _strip_templates_with_spans(text)[0]


def _strip_templates_with_spans(text: str):
    """
    Template removal.

    - A closed {{...}} (outermost; nested ones are handled inside it) is removed completely if its interior contains
      a "|" or an "=" (structured: a citation or infobox) or has at most 8 words (short).
    - A closed template with none of these (long, no "|", no "=") is really prose that was wrapped in braces, e.g.
      "{{cite web Nadiadwala made his directorial debut ... }}": only its "{{", its "}}" and its leading name
      ("cite"/"Cite" plus a following lowercase word) are removed and the prose is kept. Templates nested inside it
      are judged by the same rule.
    - An unclosed "{{" (no matching "}}") is not allowed to swallow the rest of the chunk: only its "{{name|" opener
      is removed.
    - A stray "}}" is dropped.

    :return: (text without templates, list of the removed substrings)
    """
    stack = []
    matched = []  # (start, end) of closed templates
    stray = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            stack.append(i)
            i += 2
        elif text.startswith("}}", i):
            if stack:
                matched.append((stack.pop(), i + 2))
            else:
                stray.append((i, i + 2))
            i += 2
        else:
            i += 1

    # nesting tree: parent of each closed template
    matched.sort(key=lambda se: (se[0], -se[1]))
    children = {None: []}
    open_parents = []
    for se in matched:
        while open_parents and open_parents[-1][1] <= se[0]:
            open_parents.pop()
        children.setdefault(se, [])
        children[open_parents[-1] if open_parents else None].append(se)
        open_parents.append(se)

    removals = []  # (start, end), non-overlapping

    def decide(se):
        start, end = se
        interior = text[start + 2:end - 2]
        if "|" in interior or "=" in interior or len(interior.split()) <= _MAX_STRUCTURELESS_WORDS:
            removals.append(se)
            return
        m = _CITE_NAME_RE.match(text, start + 2)
        removals.append((start, m.end() if m else start + 2))
        removals.append((end - 2, end))
        for child in children[se]:
            decide(child)

    for se in children[None]:
        decide(se)

    def covered(pos):
        return any(s <= pos < e for s, e in removals)

    for start in stack:  # unclosed openers
        if not covered(start):
            m = _OPENER_RE.match(text, start)
            removals.append((start, m.end() if m else start + 2))
    removals.extend(stray)
    removals.sort()
    out, spans, pos = [], [], 0
    for start, end in removals:
        out.append(text[pos:start])
        out.append(" ")
        spans.append(text[start:end])
        pos = end
    out.append(text[pos:])
    return "".join(out), spans


def _is_heading(fragment: str) -> bool:
    """True if a split-off fragment looks like a section heading ("Early life.", "Club career.:")."""
    stripped = fragment.strip()
    if not stripped:
        return False
    had_colon = stripped.endswith(".:") or stripped.endswith(":")
    core = stripped.rstrip(".:").strip()
    words = core.split()
    if not words or len(words) > 5 or any(ch.isdigit() for ch in core):
        return False
    if core.lower() in _HEADING_PHRASES or words[-1].lower() in _HEADING_LAST_WORDS:
        return True
    # ".:" is how the chunker renders heading boundaries; accept short capitalised phrases with it
    return had_colon and len(words) <= 4 and words[0][0].isupper()


def split_into_sentences(text: str) -> List[str]:
    """
    Regex sentence splitter (no punkt / spaCy).

    Splits on . ! ? followed by whitespace and a capital letter, digit or opening quote/bracket, but not after
    abbreviations (Mr., Dr., St., Jr., Prof., Inc. ...), initials (J. K. Rowling), dotted acronyms (U.S.),
    "c. 1950" / "No. 5" style abbreviations before numbers, and decimals (3.5). Section-heading prefixes such as
    "Early life." or "Club career.:" are split off and discarded rather than glued onto the next sentence.

    :param text: (ideally cleaned) passage text
    :return: list of sentence strings, in order, without empty entries or headings
    """
    if not text or not text.strip():
        return []
    text = _WS_RE.sub(" ", text).strip()

    for pattern in (_ALWAYS_PROTECT, _BEFORE_NUMBER, _BEFORE_LOWER):
        text = pattern.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    text = _INITIAL.sub(lambda m: m.group(1) + _PLACEHOLDER, text)
    text = _DOTTED.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)
    text = _INLINE_DOTTED.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER), text)

    pieces = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        pieces.append(text[start:m.end(1)])
        start = m.end()
    pieces.append(text[start:])

    sentences = []
    for piece in pieces:
        piece = piece.replace(_PLACEHOLDER, ".").strip()
        if not piece or _is_heading(piece):
            continue
        sentences.append(piece)
    return sentences


def _truncate_words(text: str, max_words: int) -> str:
    """Keeps at most max_words whitespace-separated words."""
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def build_premises(passages: List[dict], window: int = 2, max_words: int = 180) -> List[Tuple[str, int]]:
    """
    Builds candidate premises from retrieved passages using sliding windows of sentences.

    Each passage is cleaned and split into sentences; every run of `window` consecutive sentences (stride 1) becomes
    a premise (a passage with fewer sentences yields one premise of all of them). A window longer than max_words is
    shortened by dropping its trailing sentences, and if its first sentence alone is too long it is truncated to
    max_words words. Premises are deduplicated (case- and whitespace-insensitive) across all passages, keeping the
    first occurrence.

    :param passages: list of dicts with a "text" key (as in FactExample.passages)
    :param window: number of consecutive sentences per premise
    :param max_words: maximum words per premise
    :return: list of (premise, passage_index) in passage order, where passage_index indexes into `passages`
    """
    window = max(1, window)
    premises: List[Tuple[str, int]] = []
    seen = set()
    for idx, passage in enumerate(passages):
        sentences = split_into_sentences(clean_passage_text(passage.get("text", "")))
        if not sentences:
            continue
        for j in range(max(1, len(sentences) - window + 1)):
            chunk = sentences[j:j + window]
            while len(chunk) > 1 and sum(len(s.split()) for s in chunk) > max_words:
                chunk = chunk[:-1]
            premise = _truncate_words(" ".join(chunk), max_words)
            key = _WS_RE.sub(" ", premise).casefold()
            if key in seen:
                continue
            seen.add(key)
            premises.append((premise, idx))
    return premises


def build_pairs(fact: str, passages: List[dict], window: int = 2, max_words: int = 180) -> List[dict]:
    """
    Pairs every premise from build_premises with the fact, ready for an NLI model.

    :param fact: the fact to check; whitespace is normalised (runs collapsed, ends stripped) to form the hypothesis
    :param passages: list of dicts with a "text" key
    :param window: sentences per premise (see build_premises)
    :param max_words: maximum words per premise (see build_premises)
    :return: list of {"premise": str, "hypothesis": str, "passage_index": int}, in build_premises order
    """
    hypothesis = _WS_RE.sub(" ", fact or "").strip()
    return [{"premise": premise, "hypothesis": hypothesis, "passage_index": idx}
            for premise, idx in build_premises(passages, window=window, max_words=max_words)]


_ALPHA_WORD_RE = re.compile(r"[^\W\d_]+")


def coverage_ratio(passages: List[dict]) -> float:
    """
    Fraction of alphabetic words that survive clean_passage_text, measured against the text AFTER markup removal.

    Caveat: markers, templates, URLs, key=value residue, URL fragments and U+FFFD words are removed from the denominator with the same code that
    cleans the text, so this ratio does NOT detect markup-removal mistakes (e.g. a template that swallows real
    sentences). In practice it only measures the loss from dropping a repeated leading title. Use the
    removed_span_words / swallowed_sentences fields of validate_example for template-removal problems. Words are
    compared as lower-cased multisets, per passage.

    :param passages: list of dicts with a "text" key
    :return: ratio in [0, 1]; 1.0 if there are no countable words
    """
    from collections import Counter
    expected_total = kept_total = 0
    for passage in passages or []:
        original = (passage.get("text") if isinstance(passage, dict) else None) or ""
        expected = Counter(w.lower() for w in _ALPHA_WORD_RE.findall(_remove_markup(original)))
        cleaned = Counter(w.lower() for w in _ALPHA_WORD_RE.findall(clean_passage_text(original)))
        expected_total += sum(expected.values())
        kept_total += sum(min(c, cleaned[w]) for w, c in expected.items())
    return kept_total / expected_total if expected_total else 1.0


def validate_example(example) -> dict:
    """
    Sanity-checks the preprocessing of one FactExample. Never raises: on an unexpected error the returned dict
    carries an "error" string and the remaining fields keep their defaults.

    :param example: a FactExample (needs .passages, a list of dicts with a "text" key)
    :return: dict with n_passages, n_empty_passages (passages with no sentence after cleaning), n_sentences,
             n_premises, max_premise_words, empty_after_cleaning (True if no premise at all),
             leftover_markup (any of "{{", "}}", "<s>", "</s>", "http" in any premise) and
             tokens_lost (any whitespace-separated word of a cleaned sentence that does not occur in the original
             passage text),
             coverage (coverage_ratio of the example's passages), removed_span_words (words in the longest single
             span removed by template stripping) and swallowed_sentences (True if any removed span contains a
             sentence boundary: a period, a space and a capital letter)
    """
    result = {
        "n_passages": 0, "n_empty_passages": 0, "n_sentences": 0, "n_premises": 0, "max_premise_words": 0,
        "empty_after_cleaning": True, "leftover_markup": False, "tokens_lost": False, "coverage": 1.0,
        "removed_span_words": 0, "swallowed_sentences": False,
    }
    try:
        passages = getattr(example, "passages", None) or []
        result["n_passages"] = len(passages)
        for passage in passages:
            original = (passage.get("text") if isinstance(passage, dict) else None) or ""
            sentences = split_into_sentences(clean_passage_text(original))
            _, spans = _strip_templates_with_spans(original.replace("<s>", " ").replace("</s>", " "))
            for span in spans:
                result["removed_span_words"] = max(result["removed_span_words"], len(span.split()))
                if _SENTENCE_BOUNDARY_RE.search(span):
                    result["swallowed_sentences"] = True
            if not sentences:
                result["n_empty_passages"] += 1
                continue
            result["n_sentences"] += len(sentences)
            if not result["tokens_lost"]:
                result["tokens_lost"] = any(w not in original for s in sentences for w in s.split())
        result["coverage"] = coverage_ratio(passages)
        premises = build_premises(passages)
        result["n_premises"] = len(premises)
        result["empty_after_cleaning"] = not premises
        result["max_premise_words"] = max((len(p.split()) for p, _ in premises), default=0)
        markers = ("{{", "}}", "<s>", "</s>", "http")
        result["leftover_markup"] = any(m in p for p, _ in premises for m in markers)
    except Exception as e:  # never raise
        result["error"] = f"{type(e).__name__}: {e}"
    return result
