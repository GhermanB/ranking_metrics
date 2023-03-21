from typing import List
import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    score = np.sum(relevance[:k])
    return float(score)


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = 0

    if method == "standard":
        for idx, i in enumerate(relevance[:k]):
            one_element_score = i / np.log2(idx + 2)
            score += one_element_score

    elif method == "industry":
        for idx, i in enumerate(relevance[:k]):
            one_element_score = ((2 ** i) - 1) / np.log2(idx + 2)
            score += one_element_score

    return score


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    sorted_relevance = list(sorted(relevance, reverse=True))
    dcg = 0
    idcg = 0

    if method == "standard":
        for idx, i in enumerate(sorted_relevance[:k]):
            one_element_score = i / np.log2(idx + 2)
            idcg += one_element_score
        for idx, i in enumerate(relevance[:k]):
            one_element_score = i / np.log2(idx + 2)
            dcg += one_element_score

    elif method == "industry":
        for idx, i in enumerate(sorted_relevance[:k]):
            one_element_score = (2 ** i - 1) / np.log2(idx + 2)
            idcg += one_element_score
        for idx, i in enumerate(relevance[:k]):
            one_element_score = (2 ** i - 1) / np.log2(idx + 2)
            dcg += one_element_score

    else:
        raise ValueError

    score = dcg / idcg
    return score


def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """avarage nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    average_ndcg = []
    for relevance in list_relevances:
        sorted_relevance = list(sorted(relevance[:k], reverse=True))
        dcg = 0
        idcg = 0

        if method == "standard":
            for idx, i in enumerate(sorted_relevance):
                one_element_score = i / np.log2(idx + 2)
                idcg += one_element_score
            for idx, i in enumerate(relevance[:k]):
                one_element_score = i / np.log2(idx + 2)
                dcg += one_element_score

        elif method == "industry":
            for idx, i in enumerate(sorted_relevance):
                one_element_score = (2 ** i - 1) / np.log2(idx + 2)
                idcg += one_element_score
            for idx, i in enumerate(relevance[:k]):
                one_element_score = (2 ** i - 1) / np.log2(idx + 2)
                dcg += one_element_score

        else:
            raise ValueError

        average_ndcg.append(dcg / idcg)

    print(average_ndcg)
    score = sum(average_ndcg) / len(list_relevances)
    return score


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """recall_k"""
    k_items = sorted(list(zip(scores, labels)), reverse=True)[:k]
    _, k_labels = zip(*k_items)
    recall_k = sum(k_labels) / sum(labels)

    return recall_k


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """precision_k"""
    k_items = sorted(list(zip(scores, labels)), reverse=True)[:k]
    _, k_labels = zip(*k_items)
    precision_k = sum(k_labels) / len(k_labels)
    return precision_k


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """specificity"""
    sorted_items = sorted(list(zip(scores, labels)), reverse=True)
    _, sorted_labels = zip(*sorted_items)
    true_n = sorted_labels[k:].count(0)
    negatives = labels.count(0)
    if negatives == 0:
        return 0
    else:
        specificity_k = true_n / negatives
        return specificity_k


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """f1"""
    sorted_items = sorted(list(zip(scores, labels)), reverse=True)
    _, sorted_labels = zip(*sorted_items)
    tp = sorted_labels[:k].count(1)
    fp = sorted_labels[:k].count(0)
    fn = sorted_labels[k:].count(1)
    f1_k = 2 * tp / (2 * tp + fp + fn)
    return f1_k
