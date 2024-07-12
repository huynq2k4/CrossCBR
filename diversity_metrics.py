import torch
import numpy as np
from scipy.stats import entropy
import sys


def diversity_calculator(rank_list, num_item):
    rank_list = rank_list.long()
    freq_item = np.full(num_item, 0.0, dtype=float)

    for b in range(rank_list.size(0)):
        for i in range(rank_list.size(1)):
            freq_item[rank_list[b, i].item()] += 1.0

    freq_item = freq_item / np.sum(freq_item)
    freq_item = np.sort(freq_item)
    entropy_aggregate = entropy(freq_item, base=2)
    gini_vector = [(2 * i - num_item + 1) * freq_item[i] for i in range(num_item)]
    gini = np.sum(gini_vector) / (num_item - 1)
    
    metrics = {
        'entropy_aggregate': entropy_aggregate,
        'gini': gini
    }
    return metrics