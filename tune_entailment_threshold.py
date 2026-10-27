import argparse
import json
import os
import re
import gc
from nltk.stem import PorterStemmer
from passage_prep import build_premises

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
SCORES_PATH = "entailment_scores.jsonl"


def load_saved_scores():
    scores = []

    with open(SCORES_PATH, "r", encoding="utf-8") as file:
        for line in file:
            scores.append(json.loads(line))

    return scores


def save_scores(scores):
    with open(SCORES_PATH, "w", encoding="utf-8") as file:
        for item in scores:
            file.write(json.dumps(item) + "\n")


def calculate_scores(use_cuda):
    print("Loading development data...")

    fact_to_passage_dict = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, fact_to_passage_dict)

    print(f"Loaded {len(examples)} examples.")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-enabled GPU is available."
            )
        model.to("cuda")

    model.eval()
    ent_model = EntailmentModel(model, tokenizer, use_cuda)

    scores = []
    stemmer = PorterStemmer()

    def _tokenize(text: str) -> set:
        text = text.replace("<s>", " ").replace("</s>", " ")
        return {stemmer.stem(token) for token in re.findall(r"\w+", text.lower())}

    for example in tqdm(examples, desc="Scoring facts"):
        max_score = 0.0
        any_entailment = False
        fact_tokens = _tokenize(example.fact)

        if fact_tokens:
            # Chunk into sentence windows just like factcheck.py
            premises = build_premises(example.passages, window=2)

            for premise_text, _ in premises:
                premise_tokens = _tokenize(premise_text)

                # Word overlap pruning
                if premise_tokens:
                    overlap_count = sum(1 for token in fact_tokens if token in premise_tokens)
                    word_recall = overlap_count / len(fact_tokens)
                else:
                    word_recall = 0.0

                if word_recall < 0.08:
                    continue

                # Run entailment
                score, predicted_class = ent_model.check_entailment(
                    premise_text, example.fact
                )

                max_score = max(max_score, score)

                if predicted_class == 0:
                    any_entailment = True

        gold_label = "NS" if example.label == "IR" else example.label

        scores.append({
            "fact": example.fact,
            "gold_label": gold_label,
            "max_entailment_score": float(max_score),
            "discrete_prediction": "S" if any_entailment else "NS",
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    save_scores(scores)
    print(f"Saved scores to {SCORES_PATH}")

    return scores


def evaluate_thresholds(scores):
    results = []

    for step in range(5, 96):
        threshold = step / 100
        correct = 0

        for item in scores:
            prediction = (
                "S"
                if item["max_entailment_score"] >= threshold
                else "NS"
            )

            if prediction == item["gold_label"]:
                correct += 1

        accuracy = correct / len(scores)
        results.append((accuracy, threshold, correct))

    results.sort(reverse=True)

    print("\nTop 10 thresholds on the development set:")
    print("Threshold | Correct | Accuracy")
    print("-" * 34)

    for accuracy, threshold, correct in results[:10]:
        print(
            f"{threshold:9.2f} | {correct:7d} | {accuracy:.4%}"
        )

    best_accuracy, best_threshold, best_correct = results[0]

    print(f"\nBest development threshold: {best_threshold:.2f}")
    print(
        f"Development accuracy: "
        f"{best_correct}/{len(scores)} = {best_accuracy:.4%}"
    )
    print(
        "\nNote: the threshold is selected on the development set. "
        "Evaluate the final choice according to the coursework protocol."
    )


def evaluate_discrete_strategy(scores):
    correct = sum(
        item["discrete_prediction"] == item["gold_label"]
        for item in scores
    )

    total = len(scores)

    print("\nDiscrete strategy results:")
    print(f"Accuracy: {correct}/{total} = {correct / total:.4%}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reuse-scores",
        action="store_true",
        help="Reuse saved scores without loading the model.",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA if a compatible GPU is available.",
    )

    args = parser.parse_args()

    if args.reuse_scores:
        if not os.path.exists(SCORES_PATH):
            raise FileNotFoundError(
                f"{SCORES_PATH} does not exist. "
                "Run without --reuse-scores first."
            )

        print(f"Reusing saved scores from {SCORES_PATH}")
        scores = load_saved_scores()

        # Older score files do not contain discrete predictions.
        if any("discrete_prediction" not in item for item in scores):
            print(
                "Saved scores lack discrete predictions. "
                "Run without --reuse-scores to regenerate them."
            )
            return
    else:
        scores = calculate_scores(args.cuda)

    evaluate_thresholds(scores)
    evaluate_discrete_strategy(scores)


if __name__ == "__main__":
    main()
