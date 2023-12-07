import nltk
import pandas as pd
import streamlit as st
from scores import Meteor, Bleu, Level

st.set_page_config("Score Generator", page_icon=":gear:")


with st.sidebar:
    st.header("Settings")
    st.checkbox("Show Scores As Percentage", False, key="show_scores_as_percentage")
    level = st.radio(
        "Level",
        list(Level),
        help=Level.__doc__,
        format_func=lambda x: str.title(x.name),
    )

    assert level is not None

    with st.expander("Meteor Parameters"):
        meteor = Meteor(level)
        meteor.get_parameters()
    with st.expander("BLEU Parameters"):
        bleu = Bleu(level)
        bleu.get_parameters()


if level == Level.SENTENCE:
    references = st.session_state.setdefault("sentence_references", [])
    hypothesis = st.session_state.setdefault("sentence_hypothesis", [])
    reference_text_input_column, candidate_text_input_column = st.columns(2)
    with reference_text_input_column:
        reference_text_area = st.text_area("References")
    with candidate_text_input_column:
        candidate_text_input = st.text_input("Prediction")

    if (not reference_text_area or not candidate_text_input) and (
        not references or not hypothesis
    ):
        st.warning("Please fill in both of the provided fields.")
        st.stop()

    df = pd.DataFrame(
        {
            "References": [reference_text_area.splitlines()],
            "Candidate": candidate_text_input,
        }
    )
    edited_df = st.data_editor(df, use_container_width=True)
    references = st.session_state["sentence_references"] = [
        reference_text.split() for reference_text in reference_text_area.splitlines()
    ]
    hypothesis = st.session_state["sentence_hypothesis"] = (
        edited_df["Candidate"].tolist()[0].split()
    )

elif level == Level.CORPUS:
    references = st.session_state.setdefault("corpus_references", [])
    hypothesis = st.session_state.setdefault("corpus_hypothesis", [])
    with st.form("corpus_level_text_input", clear_on_submit=True):
        reference_text_input_column, candidate_text_input_column = st.columns(2)
        with reference_text_input_column:
            reference_text_area = st.text_area("References")
        with candidate_text_input_column:
            candidate_text_input = st.text_input("Prediction")
        add_text_button = st.form_submit_button(
            "Add To Corpus", use_container_width=True
        )

        if (not reference_text_area or not candidate_text_input) and (
            not references or not hypothesis
        ):
            st.toast("Please fill in both of the provided fields.", icon="⚠️")
            st.stop()
        if add_text_button:
            references.append(
                [
                    reference_text.split()
                    for reference_text in reference_text_area.splitlines()
                ]
            )
            hypothesis.append(candidate_text_input.split())

    with st.expander("Text in Corpus"):
        df = pd.DataFrame(
            {
                "References": [
                    [" ".join(reference) for reference in list_of_references]
                    for list_of_references in references
                ],
                "Candidates": [" ".join(hypo) for hypo in hypothesis],
            }
        )
        edited_df = st.data_editor(
            df,
            num_rows="fixed",
            column_config={"References": st.column_config.ListColumn("References")},
            use_container_width=True,
            key="edited_df",
        )
        hypothesis = st.session_state["corpus_hypothesis"] = [
            hypo.split() for hypo in edited_df["Candidates"].tolist()
        ]

        clear_corpus_button = st.button("Clear Corpus", use_container_width=True)
        if clear_corpus_button:
            st.session_state["corpus_references"] = []
            st.session_state["corpus_hypothesis"] = []
            st.stop()


nltk.download("wordnet")
meteor_score_column, bleu_score_column = st.columns(2)

with meteor_score_column:
    meteor.show_score(references, hypothesis)
with bleu_score_column:
    bleu.show_score(references, hypothesis)

st.divider()

with st.expander("Metric Descriptions"):
    tab1, tab2 = st.tabs(["Meteor", "BLEU"])
    with tab1:
        st.caption("the following information is copied from https://huggingface.co/spaces/evaluate-metric/meteor")
        st.markdown(
            """
METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision.

METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.

## Inputs

METEOR has two mandatory arguments:

    predictions: a list of predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: a list of references (in the case of one reference per prediction), or a list of lists of references (in the case of multiple references per prediction. Each reference should be a string with tokens separated by spaces.

It also has several optional parameters:

    alpha: Parameter for controlling relative weights of precision and recall. The default value is 0.9.
    beta: Parameter for controlling shape of penalty as a function of fragmentation. The default value is 3.
    gamma: The relative weight assigned to fragmentation penalty. The default is 0.5.

Refer to the [METEOR paper](https://aclanthology.org/W05-0909.pdf) for more information about parameter values and ranges.

## Limitations and bias

While the correlation between METEOR and human judgments was measured for Chinese and Arabic and found to be significant, further experimentation is needed to check its correlation for other languages.

Furthermore, while the alignment and matching done in METEOR is based on unigrams, using multiple word entities (e.g. bigrams) could contribute to improving its accuracy – this has been proposed in more recent publications on the subject."""
        )
    with tab2:
        st.caption("the following information is copied from https://huggingface.co/spaces/evaluate-metric/bleu")
        st.write(
            """
BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine’s output and that of a human: “the closer a machine translation is to a professional human translation, the better it is” – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.

Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole corpus to reach an estimate of the translation’s overall quality. Neither intelligibility nor grammatical correctness are not taken into account.

## Inputs
This metric takes as input a list of predicted sentences and a list of lists of reference sentences (since each predicted sentence can have multiple references):

    predictions (list of strs): Translations to score.
    references (list of lists of strs): references for each translation.


BLEU’s output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score.

Refer to the [BLEU paper](https://aclanthology.org/P02-1040/) for more information about the metric.

## Limitations and Bias

This metric has multiple known limitations:

- BLEU compares overlap in tokens from the predictions and references, instead of comparing meaning. This can lead to discrepancies between BLEU scores and human ratings.
- Shorter predicted translations achieve higher scores than longer ones, simply due to how the score is calculated. A brevity penalty is introduced to attempt to counteract this.
- BLEU scores are not comparable across different datasets, nor are they comparable across different languages.
- BLEU scores can vary greatly depending on which parameters are used to generate the scores, especially when different tokenization and normalization techniques are used. It is therefore not possible to compare BLEU scores generated using different parameters, or when these parameters are unknown. For more discussion around this topic, see the following issue.

"""
        )

# explanation_selectbox = st.selectbox("See Explanation", ["Meteor", "Bleu"])
# with st.expander(f"Explanation ({explanation_selectbox} Score)"):
#     if explanation_selectbox == "Bleu":
#         bleu.show_explanation(references, hypothesis)
#     elif explanation_selectbox == "Meteor":
#         meteor.show_explanation(references, hypothesis)
