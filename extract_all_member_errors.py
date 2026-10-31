# extract_all_member_errors.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel, EntailmentFactChecker


def main():
    passages_path = "data/passages_bm25_ChatGPT_humfacts.jsonl"
    labels_path = "data/dev_labeled_ChatGPT.jsonl"

    print("Loading development data...")
    fact_to_passage = read_passages(passages_path)
    examples = read_fact_examples(labels_path, fact_to_passage)

    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    use_cuda = torch.cuda.is_available()
    # use_cuda = False #to force run on CPU
    if use_cuda:
        model.to('cuda')
    model.eval()

    ent_model = EntailmentModel(model, tokenizer, use_cuda)
    checker = EntailmentFactChecker(ent_model)

    false_positives = []
    false_negatives = []

    print("Evaluating dataset to extract model errors...")
    for i, ex in enumerate(examples):
        gold = "NS" if ex.label == "IR" else ex.label
        pred = checker.predict(ex.fact, ex.passages)

        if pred == "S" and gold == "NS":
            false_positives.append({"dataset_index": i, "fact": ex.fact, "passages": ex.passages})
        elif pred == "NS" and gold == "S":
            false_negatives.append({"dataset_index": i, "fact": ex.fact, "passages": ex.passages})

    print(f"\nTotal False Positives found: {len(false_positives)}")
    print(f"Total False Negatives found: {len(false_negatives)}")

    # 10 of each type
    fp_list = false_positives[:10]
    fn_list = false_negatives[:10]

    # Member 1: 3 FP (indices 0-2), 2 FN (indices 0-1)
    # Member 2: 2 FP (indices 3-4), 3 FN (indices 2-4)
    # Member 3: 3 FP (indices 5-7), 2 FN (indices 5-6)
    # Member 4: 2 FP (indices 8-9), 3 FN (indices 7-9)

    allocations = {
        "Member 1": {"fps": fp_list[0:3], "fns": fn_list[0:2]},
        "Member 2": {"fps": fp_list[3:5], "fns": fn_list[2:5]},
        "Member 3": {"fps": fp_list[5:8], "fns": fn_list[5:7]},
        "Member 4": {"fps": fp_list[8:10], "fns": fn_list[7:10]}
    }

    for member, data in allocations.items():
        print(f"\n{'=' * 20} {member.upper()} ASSIGNED ERRORS {'=' * 20}")

        print(f"--- False Positives (Model = S, Gold = NS) [{len(data['fps'])} samples] ---")
        for idx, item in enumerate(data["fps"], 1):
            print(f"  [FP #{idx}] Dataset Index: {item['dataset_index']}")
            print(f"    Fact: {item['fact']}")

        print(f"--- False Negatives (Model = NS, Gold = S) [{len(data['fns'])} samples] ---")
        for idx, item in enumerate(data["fns"], 1):
            print(f"  [FN #{idx}] Dataset Index: {item['dataset_index']}")
            print(f"    Fact: {item['fact']}")


if __name__ == "__main__":
    main()