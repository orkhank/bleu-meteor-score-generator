# Score Generator Streamlit App

## Overview

This Streamlit app provides a user interface for generating and visualizing scores for natural language processing tasks. The app supports two scoring levels: Sentence Level and Corpus Level, and currently implements two scoring metrics: METEOR and BLEU.

## Getting Started

To run the app, make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Once you have the dependencies installed, run the app by executing the following command in your terminal:

```bash
streamlit run app.py
```

## Usage

### Input

The app allows you to input reference and hypothesis text for scoring. Depending on the selected scoring level (Sentence or Corpus), you can provide multiple references for each hypothesis.

#### Sentence Level Scoring

- For each sentence, enter reference(s) and the candidate hypothesis.
- Click on the "Add To Corpus" button to add the input to the corpus.

#### Corpus Level Scoring

- Enter reference(s) and the candidate hypothesis for the entire corpus.
- Click on the "Add To Corpus" button to add the input to the corpus.

### Settings

Use the settings on the sidebar to configure the scoring parameters and choose the scoring level:

- **Show Scores As Percentage:** Toggle to display scores as percentages.
- **Level:** Choose between Sentence and Corpus level scoring.

### METEOR Parameters

Expand the "Meteor Parameters" section to set parameters for METEOR scoring:

- **Alpha:** Control the relative weights of precision and recall.
- **Beta:** Control the shape of penalty as a function of fragmentation.
- **Gamma:** Relative weight assigned to the fragmentation penalty.

### BLEU Parameters

Expand the "BLEU Parameters" section to set parameters for BLEU scoring:

- **Max Order of N-Grams:** Set the maximum order of N-grams.
- **Weights:** Set weights for each N-gram.
- **Smoothing Function:** Toggle to enable smoothing of harsh scores.
- **Auto Reweigh:** Toggle to re-normalize weights uniformly.

### Metric Descriptions

Expand the "Metric Descriptions" section to view detailed descriptions of METEOR and BLEU metrics.

### Score Display

The app displays scores for both METEOR and BLEU in separate columns. The scores can be shown as percentages or raw values.

## Explanation

For Sentence level scoring, the app provides an explanation of how BLEU brevity penalty is calculated. An explanation for Corpus level scoring is planned for future implementation.

## References

- METEOR: [Lavie-Agarwal-2007-METEOR.pdf](https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf)
- BLEU: [Papineni et al. (2002)](https://www.aclweb.org/anthology/P02-1040.pdf)
