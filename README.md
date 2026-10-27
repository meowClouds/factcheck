# Deep Learning Fact-Checking Pipeline (CM3604)

This repository contains a fact-checking pipeline designed to evaluate whether a given factual statement is Supported (S) or Not Supported (NS) by a set of retrieved Wikipedia passages. The project is split into bag-of-words overlap methods and a deep learning textual entailment model (DeBERTa-v3).

## Setup and Installation

1. Create and activate a Python 3.10 virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows Git Bash
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration on Windows with RTX GPUs, install the CUDA-enabled version of PyTorch directly from the PyTorch index.*

3. Ensure the dataset files are placed in the `data/` directory:
   * `data/dev_labeled_ChatGPT.jsonl`
   * `data/passages_bm25_ChatGPT_humfacts.jsonl`

## Execution

**Run Word Overlap Mode:**
```bash
python factchecking_main.py --mode word_overlap
```

**Run Entailment Mode (CPU):**
```bash
python factchecking_main.py --mode entailment
```

**Run Entailment Mode (GPU Accelerated):**
```bash
python factchecking_main.py --mode entailment --cuda
```

---

## Project Roadmap & Progress

### Phase 1: Part 1 - Word Overlap Implementation (Completed)
**Objective:** Implement bag-of-words overlap, handle tokenization/stemming/stop words, tune similarity metrics (TF-IDF, Jaccard, Recall), optimize thresholds, and achieve at least 75% accuracy.
*   **Lead Developers (Coding & Preprocessing - Members 1 & 2):** Handled tokenization, stop-word removal, stemming choices, and integrated scoring metrics directly into `factcheck.py`.
*   **Co-Developers & Testers (Tuning & Validation - Members 3 & 4):** Focused on threshold tuning, running experiments, and successfully verifying that the final accuracy met the >75% target.

### Phase 2: Part 2 - Textual Entailment & Pruning (Completed)
**Objective:** Implement the DeBERTa-v3 base model mapping to binary decisions ($S/NS$), sentence splitting/cleaning, and word-overlap pruning to avoid out-of-memory issues, aiming for at least 83% accuracy. 
*Current Status: Achieved 85.06% accuracy using a tuned threshold of 0.43 on 2-sentence sliding windows.*
*   **Member 3 (Passage Cleaning & Sentence Splitting Lead):** Implemented text cleaning and sentence-splitting techniques (`build_premises`) to break raw passage texts down into individual sentences, ensuring DeBERTa-v3 can process premise-hypothesis pairs cleanly without truncation.
*   **Member 4 (Binary Decision Mapping Lead):** Bridged the gap between the model's three-class output and the final binary result, implementing the strategy to map entailment logits into $S/NS$ decisions using max-probability thresholding.
*   **Member 1 (Word-Overlap Pruning & Memory Management Lead):** Implemented low-word-overlap pruning (threshold: 0.08) using metrics from Part 1 to filter out irrelevant instances. Monitored memory usage and integrated OOM prevention (`gc.collect()`, `torch.cuda.empty_cache()`).
*   **Member 2 (GPU Execution, Threshold Tuning & Accuracy Verification Lead):** Handled model queries using GPU acceleration via the `--cuda` flag, resolved PyTorch architecture mismatches, tuned the classification threshold to 0.43, and verified the final setup cleared the 83% accuracy target (achieving 85.06%).

### Phase 3: Part 3 - Error Analysis & Written Report (In Progress)
**Objective:** Examine 10 false positives and 10 false negatives, establish 2–4 fine-grained error categories, write definitions, compute aggregate statistics, and detail 3 specific examples. 
*   **Data Mining Leads (Error Extraction - Members 1 & 3):** Pull the model errors, isolate the 10 false positives and 10 false negatives, and log their text and labels.
*   **Categorization & Writing Leads (Members 2 & 4):** Draft the fine-grained error category definitions, compile aggregate statistics, and write up the 3 detailed examples.

### Phase 4: Integration, Final Testing, and Submission (Upcoming)
**Objective:** Ensure the code runs cleanly without errors via command line and package the final PDF write-up and `factcheck.py`.
*   **Integration Leads:** Member 1 & Member 3.
*   **Proofreading & Formatting Leads:** Member 2 & Member 4.