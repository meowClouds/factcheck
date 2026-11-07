# factcheck.py

import torch
from typing import List, Tuple
import numpy as np
import spacy
import gc
import re
from nltk.stem import PorterStemmer


# ---------------------------------------------------------------------------------------------
# Inlined from passage_prep.py so that factcheck.py is self-contained (standard library only).
# ---------------------------------------------------------------------------------------------
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

class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise,hypothesis,return_tensors="pt",truncation=True,max_length=512,padding=True,)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        probabilities = torch.softmax(logits, dim=-1)[0]
        entailment_score = probabilities[0].item()
        predicted_class = torch.argmax(probabilities).item()

        del inputs, outputs, logits, probabilities
        gc.collect()

        return entailment_score, predicted_class

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """
    Word-overlap fact checker: predicts "S" when enough of the fact's words appear in the retrieved passages.

    - Metric: plain word recall (fraction of the fact's tokens found in the passages). It matched or beat
      IDF-weighted recall in 5-fold CV and clearly beat TF-IDF cosine and Jaccard, likely because those
      penalise the length of the long passages.
    - Preprocessing: Porter stemming, no stop-word removal. Removing stop words lowered cross-validated
      accuracy (about 76% vs 81%).
    - Passages: all passage texts are joined, and the <s> and </s> markers are stripped before tokenizing.
    - Threshold 0.81 gives about 81% on the dev set. Accuracy is about 80-81% across thresholds 0.775 to 0.825,
      so the choice is not sensitive. The threshold was tuned on the same 221 examples (no separate test set
      is available), so treat 81% as slightly optimistic.
    """
    def __init__(self, threshold=0.81):
        self.threshold = threshold
        self.stemmer = PorterStemmer()

    def tokenize(self, text: str) -> List[str]:
        text = text.replace("<s>", " ").replace("</s>", " ")
        return [self.stemmer.stem(token) for token in re.findall(r"\w+", text.lower())]

    def predict(self, fact: str, passages: List[dict]) -> str:
        passage_text = " ".join([p.get("text", "") for p in passages])
        passage_tokens = set(self.tokenize(passage_text))

        fact_tokens = self.tokenize(fact)
        if not fact_tokens:
            score = 0.0
        else:
            score = sum(1 for token in fact_tokens if token in passage_tokens) / len(fact_tokens)

        return "S" if score >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    """
    Entailment-based fact checker enhanced with word-overlap pruning 
    to improve execution speed and out-of-memory (OOM) prevention.
    """
    def __init__(self, ent_model, overlap_threshold=0.08):
        self.ent_model = ent_model
        self.overlap_threshold = overlap_threshold
        self.stemmer = PorterStemmer()

    def _tokenize(self, text: str) -> set:
        """Helper to tokenize and stem text for word overlap comparison."""
        text = text.replace("<s>", " ").replace("</s>", " ")
        return {self.stemmer.stem(token) for token in re.findall(r"\w+", text.lower())}

    def predict(self, fact: str, passages: List[dict]) -> str:
            max_entailment_score = 0.0
            fact_tokens = self._tokenize(fact)

            if not fact_tokens:
                return "NS"

            # Split long passages into 2-sentence windows to prevent transformer truncation
            premises = build_premises(passages, window=2)

            for premise_text, _ in premises:
                premise_tokens = self._tokenize(premise_text)

                # 1. Word-Overlap Pruning Step:
                if premise_tokens:
                    overlap_count = sum(1 for token in fact_tokens if token in premise_tokens)
                    word_recall = overlap_count / len(fact_tokens)
                else:
                    word_recall = 0.0

                # Skip the heavy transformer forward pass if word overlap is too low
                if word_recall < self.overlap_threshold:
                    continue

                # 2. Run entailment inference on the clean, short premise
                score, _ = self.ent_model.check_entailment(premise_text, fact)
                max_entailment_score = max(max_entailment_score, score)

            # 3. Out-of-Memory (OOM) Prevention
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # With clean sentence windows, true entailment scores are much higher.
            threshold = 0.43
            return "S" if max_entailment_score >= threshold else "NS"

# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
