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

Furthermore, while the alignment and matching done in METEOR is based on unigrams, using multiple word entities (e.g. bigrams) could contribute to improving its accuracy – this has been proposed in more recent publications on the subject.
