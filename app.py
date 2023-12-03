import nltk
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
    col1, col2 = st.columns(2)
    with col1:
        reference_text_area = st.text_area("Reference Text")
    with col2:
        candidate_text_input = st.text_input("Candidate Text")

    if not reference_text_area or not candidate_text_input:
        st.warning("Please fill in both of the provided fields.")
        st.stop()

    references = [
        reference_text.split() for reference_text in reference_text_area.splitlines()
    ]
    hypothesis = candidate_text_input.split()
elif level == Level.CORPUS:
    st.warning("To Be Implemented...")
    st.stop()
    pass
else:
    raise ValueError


nltk.download("wordnet")
meteor_score_column, bleu_score_column = st.columns(2)

with meteor_score_column:
    meteor.show_score(references, hypothesis)
with bleu_score_column:
    bleu.show_score(references, hypothesis)

st.divider()

explanation_selectbox = st.selectbox("See Explanation", ["Meteor", "Bleu"])
with st.expander(f"Explanation ({explanation_selectbox} Score)"):
    if explanation_selectbox == "Bleu":
        bleu.show_explanation(references, hypothesis)
    elif explanation_selectbox == "Meteor":
        meteor.show_explanation(references, hypothesis)
