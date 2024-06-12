import numpy as np
from scipy.special import softmax
from scipy.stats import entropy


def get_inception_score(logits):
    prob = softmax(logits, axis=1)
    return np.exp(np.mean(entropy(prob, prob.mean(axis=0))))
