
import re
import sys

import torch
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel
from passage_prep import build_premises

PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
THRESHOLD = 0.43          
OVERLAP_THRESHOLD = 0.08  
CLASSES = ["entailment", "neutral", "contradiction"]

DEFAULT = [67, 105, 109, 79, 80, 2, 5, 34, 40, 51, 122, 123, 8, 9, 41, 77, 78, 99, 141, 144]

stemmer = PorterStemmer()


def tokens(text):
    text = text.replace("<s>", " ").replace("</s>", " ")
    return {stemmer.stem(t) for t in re.findall(r"\w+", text.lower())}


def main():
    idxs = [int(a) for a in sys.argv[1:]] or DEFAULT
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    ent = EntailmentModel(model, tokenizer, False)

    for i in idxs:
        ex = examples[i]
        gold = "NS" if ex.label == "IR" else ex.label
        fact_tokens = tokens(ex.fact)
        rows, pruned = [], 0
        for premise, passage_idx in build_premises(ex.passages, window=2):
            ptok = tokens(premise)
            recall = sum(t in ptok for t in fact_tokens) / len(fact_tokens) if fact_tokens and ptok else 0.0
            if recall < OVERLAP_THRESHOLD:
                pruned += 1
                continue
            score, cls = ent.check_entailment(premise, ex.fact)
            rows.append((score, cls, recall, passage_idx, premise))
        rows.sort(key=lambda r: -r[0])
        best = rows[0][0] if rows else 0.0
        pred = "S" if best >= THRESHOLD else "NS"
        emit("=" * 100)
        emit(f"IDX {i} | gold={gold} (raw label {ex.label}) | predicted={pred} | {'ERROR' if pred != gold else 'correct'}")
        emit(f"FACT: {ex.fact}")
        emit(f"max entailment score = {best:.3f} (threshold {THRESHOLD}) | premises scored = {len(rows)} | pruned = {pruned}"
             f" | premises with score >= threshold = {sum(r[0] >= THRESHOLD for r in rows)}")
        for score, cls, recall, pidx, premise in rows[:3]:
            emit(f"  score={score:.3f} argmax={CLASSES[cls]} recall={recall:.2f} passage={pidx} :: {premise[:600]}")
        emit()

    with open("explain_errors_output.txt", "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()