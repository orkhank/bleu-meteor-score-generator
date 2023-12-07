import nltk
import pandas as pd
import streamlit as st
from scores import Meteor, Bleu, Level

st.set_page_config("Score Generator", page_icon=":gear:")


with st.sidebar:
    st.header("Metric Parameters")
    st.checkbox("Show Scores As Percentage", False, key="show_scores_as_percentage")
    level = st.radio(
        "Level",
        list(Level),
        help=Level.__doc__,
        format_func=lambda x: str.title(x.name),
    )

    assert level is not None

    with st.expander("Meteor"):
        meteor = Meteor(level)
        meteor.get_parameters()
    with st.expander("BLEU"):
        bleu = Bleu(level)
        bleu.get_parameters()


if level == Level.SENTENCE:
    reference_text_input_column, candidate_text_input_column = st.columns(2)
    with reference_text_input_column:
        reference_text_area = st.text_area("Reference Text")
    with candidate_text_input_column:
        candidate_text_input = st.text_input("Candidate Text")

    if not reference_text_area or not candidate_text_input:
        st.warning("Please fill in both of the provided fields.")
        st.stop()

    df = pd.DataFrame(
        {
            "References": [reference_text_area.splitlines()],
            "Candidate": candidate_text_input,
        }
    )
    edited_df = st.data_editor(df, use_container_width=True)
    references = [
        reference_text.split() for reference_text in reference_text_area.splitlines()
    ]
    hypothesis = edited_df["Candidate"].tolist()[0].split()
elif level == Level.CORPUS:
    references = st.session_state.setdefault("corpus_references", [])
    hypothesis = st.session_state.setdefault("corpus_hypothesis", [])
    with st.form("corpus_level_text_input", clear_on_submit=True):
        reference_text_input_column, candidate_text_input_column = st.columns(2)
        with reference_text_input_column:
            reference_text_area = st.text_area("Reference Text")
        with candidate_text_input_column:
            candidate_text_input = st.text_input("Candidate Text")
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

# st.divider()

# explanation_selectbox = st.selectbox("See Explanation", ["Meteor", "Bleu"])
# with st.expander(f"Explanation ({explanation_selectbox} Score)"):
#     if explanation_selectbox == "Bleu":
#         bleu.show_explanation(references, hypothesis)
#     elif explanation_selectbox == "Meteor":
#         meteor.show_explanation(references, hypothesis)
