from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Literal, Optional
import numpy as np
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.translate.bleu_score import (
    corpus_bleu,
    sentence_bleu,
    SmoothingFunction,
    modified_precision,
    brevity_penalty,
    closest_ref_length,
)
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize


class Level(Enum):
    __doc__ = """The calculation level for scores.
- **Sentence Level:** Calculate a score for only one candidate text. There may be one or more reference text.
- **Corpus Level:** Analyze the overall quality of the candidates for the entire corpus. There may be one or more reference text for each of the candidate sentences.
"""

    SENTENCE = auto()
    CORPUS = auto()


class Score(ABC):
    @abstractmethod
    def __init__(self, level):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_score(self, references, hypothesis) -> float:
        pass

    @abstractmethod
    def show_explanation(self, references, hypothesis):
        pass

    def show_score(
        self,
        references,
        hypothesis,
        show_as_percentage: Optional[bool] = None,
    ):
        score = self.get_score(
            references,
            hypothesis,
        )

        if show_as_percentage is None:
            show_as_percentage = st.session_state["show_scores_as_percentage"]

        if show_as_percentage:
            rounded_score = f"{score*100: .1f}%"
        else:
            rounded_score = f"{score: .3f}"

        st.metric(f"{self.__class__.__name__} Score", rounded_score)


class Bleu(Score):
    def __init__(self, level: Level):
        self.level = level
        self.weights = (0.25, 0.25, 0.25, 0.25)
        self.smoothing_function = (None,)
        self.auto_reweigh: bool = False

    def get_parameters(self):
        n_weights = int(st.number_input("Max Order of N-Grams", 1, 10, 4, 1, "%d"))
        weights = []
        for i in range(1, n_weights + 1):
            weight = st.number_input(
                f"Weight of {i}-gram", 0.0, 2.0, 1 / n_weights, format="%.2f"
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
        if self.level == Level.SENTENCE:
            return sentence_bleu(
                references,
                hypothesis,
                self.weights,
                self.smoothing_function,
                self.auto_reweigh,
            )
        elif self.level == Level.CORPUS:
            return corpus_bleu(
                references,
                hypothesis,
                self.weights,
                self.smoothing_function,
                self.auto_reweigh,
            )
        else:
            raise ValueError

    def show_explanation(self, references, hypothesis):
        # TODO: finish level = "sentence" and add explanation for level = "corpus"
        if self.level == Level.SENTENCE:
            # Tokenize candidate translation and reference translations
            candidate_tokens = hypothesis
            reference_tokens = references

            # Calculate individual n-gram precisions
            individual_precisions = [
                modified_precision(reference_tokens, candidate_tokens, i)
                for i in range(1, len(self.weights) + 1)
            ]

            hyp_len = len(candidate_tokens)
            closest_ref_len = closest_ref_length(reference_tokens, hyp_len)

            # Calculate the brevity penalty
            brevity_penalty_value = brevity_penalty(closest_ref_len, hyp_len)

            st.write(
                """BLEU is computed using a couple of ngram modified precisions. Specifically,"""
            )
            st.latex(r"BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log(p_n)\right)")
            st.write(
                r"""where $$p_n$$ is the modified precision for ngram, the base of log is the natural base $$e$$, $$w_n$$ is weight between 0 and 1 for $$\log(p_n)$$ and $$\sum_{n=1}^{N}w_n=1$$, and BP is the brevity penalty to penalize short machine translations.""",
                unsafe_allow_html=True,
            )
            st.subheader("Brevity Penalty")
            st.markdown("BP is calculated with")
            st.latex(r"min(1,exp(1-\frac{reference\_length}{candidate\_length}))")
            st.markdown(
                f"""where $$reference\\_length$$ is the closest reference 
                length (word count) to $$candidate\\_length$$ (the word count of the candidate_text)."""
            )
            st.markdown(
                f"""In the example, given above, $$candidate\\_length$$ and $$reference\\_length$$
                are `{hyp_len}` and `{closest_ref_len}` respectively. Thus our brevity penalty value is"""
            )
            st.latex(
                f"""min(1,exp(1-\\frac{{{closest_ref_len}}}{{{hyp_len}}}))=\\mathbf{brevity_penalty_value:.2f}"""
            )

            st.subheader("Precisions")
            # st.markdown("precision<sub>i</sub> calculated with", unsafe_allow_html=True)
            # st.table(individual_precisions)
            st.write("<Empty>")

            st.subheader("Putting it all together")
            st.write("<Empty>")

        elif self.level == Level.CORPUS:
            st.warning("Corpus Level Explanation To Be Implemented...")
        else:
            raise ValueError


class Meteor(Score):
    def __init__(self, level: Level):
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
            help="parameter for controlling shape of penalty as a function of fragmentation.",
        )
        self.gamma = st.number_input(
            "Alpha",
            value=0.5,
            help="relative weight assigned to fragmentation penalty.",
        )

    def get_score(self, references, hypothesis):
        if self.level == Level.SENTENCE:
            return meteor_score(
                references,
                hypothesis,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
        elif self.level == Level.CORPUS:
            meteor_score_sentences_list = list()
            [
                meteor_score_sentences_list.append(meteor_score(expect, predict))
                for expect, predict in zip(references, hypothesis)
            ]
            meteor_score_res = np.mean(meteor_score_sentences_list)
            return meteor_score_res
        else:
            raise ValueError

    def show_explanation(self, references, hypothesis):
        st.warning("To be implemented...")
