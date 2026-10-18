# tune_word_overlap.py
# Experiment-only script (uses scikit-learn); sweeps preprocessing x metric and tunes the S/NS threshold.

import itertools
import math
import re
from collections import Counter

import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedKFold

from factchecking_main import read_passages, read_fact_examples

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
METRICS = ["recall", "idf_recall", "tfidf_cosine", "jaccard"]

STEMMER = PorterStemmer()
_stem_cache = {}


def clean_passage(text):
    return text.replace("<s>", " ").replace("</s>", " ")


def preprocess(text, stop, stem):
    tokens = re.findall(r"\w+", text.lower())
    if stop:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if stem:
        out = []
        for t in tokens:
            if t not in _stem_cache:
                _stem_cache[t] = STEMMER.stem(t)
            out.append(_stem_cache[t])
        tokens = out
    return tokens


def build_idf(examples, stop, stem):
    """IDF over every distinct passage in the dataset (smoothed, sklearn-style)."""
    seen = set()
    df = Counter()
    n_docs = 0
    for ex in examples:
        for p in ex.passages:
            text = clean_passage(p.get("text", ""))
            if text in seen:
                continue
            seen.add(text)
            n_docs += 1
            df.update(set(preprocess(text, stop, stem)))
    return lambda t: math.log((1 + n_docs) / (1 + df.get(t, 0))) + 1


def compute_scores(examples, stop, stem):
    idf = build_idf(examples, stop, stem)
    scores = {m: np.zeros(len(examples)) for m in METRICS}
    for i, ex in enumerate(examples):
        passage_text = " ".join(clean_passage(p.get("text", "")) for p in ex.passages)
        f_toks = preprocess(ex.fact, stop, stem)
        p_toks = preprocess(passage_text, stop, stem)
        if not f_toks:
            continue  # all scores stay 0 -> NS
        p_set, f_set = set(p_toks), set(f_toks)

        scores["recall"][i] = sum(t in p_set for t in f_toks) / len(f_toks)

        tot = sum(idf(t) for t in f_toks)
        scores["idf_recall"][i] = sum(idf(t) for t in f_toks if t in p_set) / tot

        f_tf, p_tf = Counter(f_toks), Counter(p_toks)
        fv = {t: c * idf(t) for t, c in f_tf.items()}
        pv = {t: c * idf(t) for t, c in p_tf.items()}
        dot = sum(w * pv[t] for t, w in fv.items() if t in pv)
        nf = math.sqrt(sum(w * w for w in fv.values()))
        npv = math.sqrt(sum(w * w for w in pv.values()))
        scores["tfidf_cosine"][i] = dot / (nf * npv) if nf and npv else 0.0

        union = len(f_set | p_set)
        scores["jaccard"][i] = len(f_set & p_set) / union if union else 0.0
    return scores


def best_threshold(scores, y):
    """y: 1 for S, 0 for NS. Predict S if score >= thr. Returns (thr, accuracy)."""
    uniq = np.unique(scores)
    cands = np.concatenate(([uniq[0] - 1e-6], (uniq[:-1] + uniq[1:]) / 2, [uniq[-1] + 1e-6]))
    best = (cands[0], -1.0)
    for thr in cands:
        acc = np.mean((scores >= thr) == (y == 1))
        if acc > best[1]:
            best = (thr, acc)
    return best


def cv_accuracy(scores, y, k=5, seed=0):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(scores, y):
        thr, _ = best_threshold(scores[tr], y[tr])
        accs.append(np.mean((scores[te] >= thr) == (y[te] == 1)))
    return float(np.mean(accs)), float(np.std(accs))


def threshold_sensitivity(scores, y, lo=0.50, hi=0.95, step=0.025):
    print("\nThreshold sensitivity (recall, stemming on, stop-words off):")
    print(f"{'thr':>7} {'acc':>8} {'ties':>5}")
    for thr in np.round(np.arange(lo, hi + step / 2, step), 4):
        acc = np.mean((scores >= thr - 1e-9) == (y == 1))
        ties = int(np.sum(np.abs(scores - thr) < 1e-9))
        print(f"{thr:>7.3f} {acc:>8.4f} {ties:>5}  {'#' * int(round((acc - 0.4) * 100))}")


def main():
    passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, passages)
    y = np.array([0 if ex.label in ("NS", "IR") else 1 for ex in examples])
    print(f"{len(examples)} examples, S={y.sum()}, NS={len(y) - y.sum()}, "
          f"majority baseline={max(y.mean(), 1 - y.mean()):.4f}\n")

    results = []
    for stop, stem in itertools.product([False, True], [False, True]):
        scores = compute_scores(examples, stop, stem)
        for m in METRICS:
            thr, acc = best_threshold(scores[m], y)
            cv_mean, cv_std = cv_accuracy(scores[m], y)
            results.append(dict(stop=stop, stem=stem, metric=m, thr=thr, acc=acc,
                                cv=cv_mean, cv_std=cv_std, scores=scores[m]))

    print(f"{'stop':<5} {'stem':<5} {'metric':<13} {'thr':>7} {'tuned acc':>10} {'5-fold CV acc':>16}")
    for r in results:
        print(f"{str(r['stop']):<5} {str(r['stem']):<5} {r['metric']:<13} {r['thr']:>7.4f} "
              f"{r['acc']:>10.4f} {r['cv']:>9.4f} ±{r['cv_std']:.4f}")

    best = max(results, key=lambda r: r["cv"])
    print(f"\nBest by CV accuracy: stop_words={best['stop']}, stemming={best['stem']}, "
          f"metric={best['metric']}, threshold={best['thr']:.4f}, tuned acc={best['acc']:.4f}, "
          f"CV acc={best['cv']:.4f}")
    best_tuned = max(results, key=lambda r: r["acc"])
    if best_tuned is not best:
        print(f"Best by tuned accuracy: stop_words={best_tuned['stop']}, stemming={best_tuned['stem']}, "
              f"metric={best_tuned['metric']}, threshold={best_tuned['thr']:.4f}, "
              f"tuned acc={best_tuned['acc']:.4f}, CV acc={best_tuned['cv']:.4f}")

    pred = best["scores"] >= best["thr"]
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0))); tn = int(np.sum(~pred & (y == 0)))
    print("\nConfusion matrix for best config (rows=gold, cols=pred):")
    print(f"{'':>8}{'pred S':>8}{'pred NS':>9}")
    print(f"{'gold S':>8}{tp:>8}{fn:>9}")
    print(f"{'gold NS':>8}{fp:>8}{tn:>9}")

    cfg = next(r for r in results if not r["stop"] and r["stem"] and r["metric"] == "recall")
    threshold_sensitivity(cfg["scores"], y)


if __name__ == "__main__":
    main()