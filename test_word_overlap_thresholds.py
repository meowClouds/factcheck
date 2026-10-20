import numpy as np

from factchecking_main import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"


def main():
    passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, passages)

    checker = WordRecallThresholdFactChecker()
    scores = []
    labels = []

    for example in examples:
        gold = "NS" if example.label == "IR" else example.label
        tokens = checker.tokenize(example.fact)

        passage_text = " ".join(
            p.get("text", "").replace("<s>", " ").replace("</s>", " ")
            for p in example.passages
        )
        passage_tokens = set(checker.tokenize(passage_text))

        score = (
            sum(token in passage_tokens for token in tokens) / len(tokens)
            if tokens else 0.0
        )

        scores.append(score)
        labels.append(gold)

    scores = np.array(scores)
    labels = np.array(labels)

    print(f"Examples: {len(labels)}")
    print("\nThreshold  Accuracy  S Precision  S Recall  S F1  FP  FN")

    for threshold in np.arange(0.775, 0.851, 0.0125):
        pred_s = scores >= threshold
        actual_s = labels == "S"

        tp = int(np.sum(pred_s & actual_s))
        fp = int(np.sum(pred_s & ~actual_s))
        fn = int(np.sum(~pred_s & actual_s))
        tn = int(np.sum(~pred_s & ~actual_s))

        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        print(
            f"{threshold:9.4f}  {accuracy:8.4f}  {precision:11.4f}"
            f"  {recall:8.4f}  {f1:5.4f}  {fp:2d}  {fn:2d}"
        )


if __name__ == "__main__":
    main()