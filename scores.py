from typing import Literal
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


class Bleu:
    def __init__(self, level: Literal["sentence", "corpus"] = "sentence"):
        self.level = level
        self.weights = (0.25, 0.25, 0.25, 0.25)
        self.smoothing_function = (None,)
        self.auto_reweigh: bool = False

    def get_parameters(self):
        n_weights = int(st.number_input("Max Order of N-Grams", 1, 10, 4, 1, "%d"))
        weights = []
        for i in range(1, n_weights + 1):
            weight = st.number_input(
                f"Weight of {i}-gram", 0.0, 1.0, 1 / n_weights, format="%.2f"
            )
            weights.append(weight)
        smoothing_function = st.toggle(
            "Smooting Function", False, help="Option to smooth the harsh (0) scores."
        )
        self.auto_reweigh = st.toggle(
            "Auto Reweigh", False, help="Option to re-normalize the weights uniformly."
        )

        self.weights = tuple(weights)
        self.smoothing_function = (
            SmoothingFunction().method1 if smoothing_function else None
        )

    def get_score(self, references, hypothesis):
        if self.level == "sentence":
            return sentence_bleu(
                references,
                hypothesis,
                self.weights,
                self.smoothing_function,
                self.auto_reweigh,
            )
        else:
            return corpus_bleu(
                references,
                hypothesis,
                self.weights,
                self.smoothing_function,
                self.auto_reweigh,
            )


class Meteor:
    def __init__(self, level: Literal["sentence", "corpus"] = "sentence"):
        self.level = level
        self.preprocess = str.lower
        self.stemmer = PorterStemmer()
        self.wordnet = wordnet
        self.alpha: float = 0.9
        self.beta: float = 3
        self.gamma: float = 0.5

    def get_parameters(self):
        self.alpha = st.number_input(
            "Alpha",
            value=0.9,
            help="parameter for controlling relative weights of precision and recall.",
        )
        self.beta = st.number_input(
            "Beta",
            value=3,
            help="parameter for controlling shape of penalty as a function of as a function of fragmentation.",
        )
        self.gamma = st.number_input(
            "Alpha",
            value=0.5,
            help="relative weight assigned to fragmentation penalty.",
        )

    def get_score(self, references, hypothesis):
        if self.level == "sentence":
            return meteor_score(
                references,
                hypothesis,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
        else:
            raise NotImplementedError
