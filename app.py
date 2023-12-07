import nltk
import pandas as pd
import streamlit as st
from scores import Meteor, Bleu, Level

st.set_page_config("Score Generator", page_icon=":gear:")

st.header("Score Generation")
input_container = st.container()
score_container = st.container()
st.divider()
metric_description_container = st.container()

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


with metric_description_container:
    with st.expander("# Metric Descriptions"):
        meteor_description_tab, bleu_description_tab = st.tabs(["Meteor", "BLEU"])
        with meteor_description_tab:
            st.caption(
                "the following information is copied from https://huggingface.co/spaces/evaluate-metric/meteor"
            )
            st.markdown(meteor.get_score_description())
        with bleu_description_tab:
            st.caption(
                "the following information is copied from https://huggingface.co/spaces/evaluate-metric/bleu"
            )
            st.markdown(bleu.get_score_description())

with input_container:
    if level == Level.SENTENCE:
        # Process input for sentence-level scoring
        references = st.session_state.setdefault("sentence_references", [])
        hypothesis = st.session_state.setdefault("sentence_hypothesis", [])
        reference_text_input_column, candidate_text_input_column = st.columns(2)

        # User input for references
        with reference_text_input_column:
            reference_text_area = st.text_area("References")

        # User input for candidate text
        with candidate_text_input_column:
            candidate_text_input = st.text_input("Prediction")

        # Validation and processing of input
        if (not reference_text_area or not candidate_text_input) and (
            not references or not hypothesis
        ):
            st.warning("Please fill in both of the provided fields.")
            st.stop()

        # Create a DataFrame for displaying input
        df = pd.DataFrame(
            {
                "References": [reference_text_area.splitlines()],
                "Candidate": candidate_text_input,
            }
        )

        # Editable data table for user inspection and editing
        edited_df = st.data_editor(df, use_container_width=True)
        references = st.session_state["sentence_references"] = [
            reference_text.split()
            for reference_text in reference_text_area.splitlines()
        ]
        hypothesis = st.session_state["sentence_hypothesis"] = (
            edited_df["Candidate"].tolist()[0].split()
        )

    elif level == Level.CORPUS:
        # Process input for corpus-level scoring
        references = st.session_state.setdefault("corpus_references", [])
        hypothesis = st.session_state.setdefault("corpus_hypothesis", [])

        # User input form for references and candidate text
        with st.form("corpus_level_text_input", clear_on_submit=True):
            reference_text_input_column, candidate_text_input_column = st.columns(2)

            # User input for references
            with reference_text_input_column:
                reference_text_area = st.text_area("References")

            # User input for candidate text
            with candidate_text_input_column:
                candidate_text_input = st.text_input("Prediction")

            # Button to add input to corpus
            add_text_button = st.form_submit_button(
                "Add To Corpus", use_container_width=True
            )

            # Validation and processing of input
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

        # Display and editing of corpus input
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

            # Button to clear corpus input
            clear_corpus_button = st.button("Clear Corpus", use_container_width=True)
            if clear_corpus_button:
                st.session_state["corpus_references"] = []
                st.session_state["corpus_hypothesis"] = []
                st.stop()

# Download NLTK resources
nltk.download("wordnet")
with score_container:
    meteor_score_column, bleu_score_column = st.columns(2)

    with meteor_score_column:
        meteor.show_score(references, hypothesis)  # type: ignore
    with bleu_score_column:
        bleu.show_score(references, hypothesis)  # type: ignore
