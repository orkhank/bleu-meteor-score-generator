import nltk
import streamlit as st
from scores import Meteor, Bleu

st.set_page_config("Score Generator", page_icon=":gear:")


with st.sidebar:
    st.header("Metric Parameters")
    show_scores_as_percent = st.toggle("Show Scores As Percentage", False)

    with st.expander("Meteor"):
        meteor = Meteor()
        meteor.get_parameters()
    with st.expander("BLEU"):
        bleu = Bleu()
        bleu.get_parameters()

# with st.form("Enter Test Cases Below"):
#     col1, col2 = st.columns(2)
#     with col1:
#         reference_text = st.text_area("Reference Text")

#     with col2:
#         machine_text = st.text_input("Machine Text")

#     st.form_submit_button()


col1, col2 = st.columns(2)
with col1:
    reference_text = st.text_area("Reference Text")
with col2:
    candidate_text = st.text_input("Candidate Text")

if not reference_text or not candidate_text:
    st.warning("Please fill in both of the provided fields.")
    st.stop()

references = reference_text.splitlines()
nltk.download("wordnet")
meteor_score_column, bleu_score_column = st.columns(2)
with meteor_score_column:
    st.metric(
        "Meteor Score",
        round(
            meteor.get_score(
                [reference.split() for reference in references],
                candidate_text.split(),
            ),
            4,
        ),
    )
with bleu_score_column:
    st.metric(
        "Bleu Score",
        round(
            bleu.get_score(
                [reference.split() for reference in references],
                candidate_text.split(),
            ),  # type: ignore
            4,
        ),  # type: ignore
    )
    with st.expander("Explanation (BLEU Score)"):
        bleu.get_explanation(references, candidate_text)
