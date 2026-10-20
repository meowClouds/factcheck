import json
from collections import Counter

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_passages, read_fact_examples


LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
OUTPUT_PATH = "word_overlap_errors.jsonl"


def main():
    passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, passages)
    checker = WordRecallThresholdFactChecker()

    errors = []
    counts = Counter()

    for example in examples:
        gold = "NS" if example.label == "IR" else example.label
        predicted = checker.predict(example.fact, example.passages)

        counts[(gold, predicted)] += 1

        if predicted != gold:
            passage_text = " ".join(
                passage.get("text", "")
                .replace("<s>", " ")
                .replace("</s>", " ")
                for passage in example.passages
            )

            fact_tokens = checker.tokenize(example.fact)
            passage_tokens = set(checker.tokenize(passage_text))
            overlap = (
                sum(token in passage_tokens for token in fact_tokens)
                / len(fact_tokens)
                if fact_tokens else 0.0
            )

            errors.append({
                "fact": example.fact,
                "gold": gold,
                "predicted": predicted,
                "overlap_score": round(overlap, 4),
            })

    total = len(examples)
    correct = total - len(errors)

    print(f"Examples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Accuracy: {correct / total:.4%}" if total else "Accuracy: N/A")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"{'':12} {'Pred S':>8} {'Pred NS':>8}")
    print(f"{'Actual S':12} {counts[('S', 'S')]:8} {counts[('S', 'NS')]:8}")
    print(f"{'Actual NS':12} {counts[('NS', 'S')]:8} {counts[('NS', 'NS')]:8}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        for error in errors:
            output.write(json.dumps(error, ensure_ascii=False) + "\n")

    print(f"\nSaved misclassified examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()