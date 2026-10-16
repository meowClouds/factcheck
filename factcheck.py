# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


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
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        # Change mode here to test each step:
        # "baseline"      -> No stop-word removal, no stemming/lemmatization
        # "stopword"      -> Test 1: Stop-word removal added
        # "stemming"      -> Test 2: Stop-word removal + Porter Stemming
        # "lemmatization" -> Test 3: Stop-word removal + Lemmatization
        self.mode = "baseline"

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        # Basic tokenization and lowercasing
        return re.findall(r'\w+', text.lower())

    def process_tokens(self, tokens: List[str]) -> List[str]:
        processed = []
        for token in tokens:
            # Test 1, 2, 3: Stop-word removal
            if self.mode in ["stopword", "stemming", "lemmatization"] and token in self.stop_words:
                continue

            # Test 2: Porter Stemming
            if self.mode == "stemming":
                token = self.stemmer.stem(token)

            # Test 3: Lemmatization instead of Stemming
            elif self.mode == "lemmatization":
                token = self.lemmatizer.lemmatize(token)

            processed.append(token)
        return processed

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Combine all passage text into a single string
        passage_text = " ".join([p.get("text", "") for p in passages])

        fact_tokens = self.tokenize(fact)
        passage_tokens = self.tokenize(passage_text)

        processed_fact = self.process_tokens(fact_tokens)

        processed_passage = self.process_tokens(passage_tokens)
        processed_passage_set = set(processed_passage)

        if not processed_fact:
            return "NS"

        # Metric 1: Word Recall (The original code)
        matches = sum(1 for token in processed_fact if token in processed_passage_set)
        recall_score = matches / len(processed_fact)

        # Metric 2: Jaccard Similarity
        fact_set = set(processed_fact)
        passage_set = set(processed_passage)
        intersection = len(fact_set.intersection(passage_set))
        union = len(fact_set.union(passage_set))
        jaccard_score = intersection / union if union > 0 else 0.0

        # Metric 3: TF-IDF Cosine Similarity
        fact_str = " ".join(processed_fact)
        passage_str = " ".join(processed_passage)

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([fact_str, passage_str])
            tfidf_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except ValueError:
            tfidf_score = 0.0

        score = tfidf_score

        return "S" if score >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


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
