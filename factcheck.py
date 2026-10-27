# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.stem import PorterStemmer


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

        for passage in passages:
            passage_text = passage.get("text", "")
            passage_tokens = self._tokenize(passage_text)

            # 1. Word-Overlap Pruning Step:
            # Calculate word recall overlap between the fact and the passage.
            if passage_tokens:
                overlap_count = sum(1 for token in fact_tokens if token in passage_tokens)
                word_recall = overlap_count / len(fact_tokens)
            else:
                word_recall = 0.0

            # If word overlap is below the threshold, entailment is extremely unlikely.
            # Skip the heavy transformer forward pass to save time and memory.
            if word_recall < self.overlap_threshold:
                continue

            # 2. Run entailment inference only on promising passages
            score, _ = self.ent_model.check_entailment(passage_text, fact)
            max_entailment_score = max(max_entailment_score, score)

        # 3. Out-of-Memory (OOM) Prevention:
        # Clear CUDA cache and run garbage collection after processing each example.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Probability threshold decision (tuned via development set)
        threshold = 0.20
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
