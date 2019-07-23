'''Error model based on a stochastic transducer.'''

import argparse
from collections import defaultdict
import numpy as np
from operator import itemgetter
import pynini
import tqdm

from ..lib.helper import escape_for_pynini


def dicts_to_value_pairs(dict_1, dict_2):
    '''Convert dictionaries `{ key_i: val_1_i }` and
       `{ key_i: val_2_i }` to pairs `(val_1_i, val_2_i)` for each
       `key_i`.'''
    for key in sorted(dict_1):
        if key in dict_2:
            yield (dict_1[key], dict_2[key])


def count_ngrams(strings, max_n):
    '''Count character n-grams up to max_n (including spaces) in strings.'''
    counts = defaultdict(lambda: 0)
    for string in strings:
        for i in range(len(string)):
            for j in range(min(max_n, len(string)-i)):
                counts[string[i:i+j+1]] += 1
    return dict(counts)


def merge_counters(a, b):
    result = a.copy()
    for key, val in b.items():
        if key in a:
            result[key] += val
        else:
            result[key] = val
    return result


def select_ngrams(counter, num):
    '''Select all unigrams ant most frequent n-grams of higher orders.'''
    # select the unigrams
    ngrams = [key for key in counter.keys() if len(key) <= 1]
    if len(ngrams) > num:
        raise Exception('Number of unigrams exceeds the number of allowed '
                        'n-grams.')
    # add the most frequent n-grams for n > 1
    ngrams.extend(map(
        itemgetter(0),
        sorted(((key, val) for key, val in counter.items() if len(key) > 1),
               reverse=True, key=itemgetter(1))[:num-len(ngrams)]))
    return ngrams


def string_to_ngram_ids(string, ngrams):
    '''Convert a string of length `m` to a matrix `A` of size `m*n`,
       where `n` is the maximum n-gram length. The entry `a[i,j]`
       contains the ID (index in the `ngrams` list) of the n-gram
       `string[i:i+j]` or `-1` if this ngram is not present in the
       list.'''
    max_n = max(len(ngr) for ngr in ngrams)
    ngrams_idx = { ngr : i for i, ngr in enumerate(ngrams) }
    result = -np.ones((len(string), max_n), dtype=np.int32)
    for i in range(len(string)):
        for j in range(min(max_n, len(string)-i)):
            ngr = string[i:i+j+1]
            if ngr in ngrams_idx:
                result[i,j] = ngrams_idx[ngr]
    return result


def preprocess_training_data(ocr_dict, gt_dict, max_n=3, max_ngrams=1000):
    string_pairs = dicts_to_value_pairs(ocr_dict, gt_dict)
    ocr_ngrams = count_ngrams(ocr_dict.values(), max_n)
    gt_ngrams = count_ngrams(gt_dict.values(), max_n)
    ngrams = select_ngrams(merge_counters(ocr_ngrams, gt_ngrams), max_ngrams)
    training_pairs = []
    for (ocr_str, gt_str) in string_pairs:
        training_pairs.append((
            string_to_ngram_ids(gt_str, ngrams),
            string_to_ngram_ids(ocr_str, ngrams)))
    return training_pairs, ngrams


def training_pairs_to_ngrams(training_pairs, max_n=3, max_ngrams=1000):
    ocr_ngrams = count_ngrams(map(itemgetter(0), training_pairs), max_n)
    gt_ngrams = count_ngrams(map(itemgetter(1), training_pairs), max_n)
    ngrams = select_ngrams(merge_counters(ocr_ngrams, gt_ngrams), max_ngrams)
    ngr_training_pairs = []
    for (ocr_str, gt_str) in training_pairs:
        ngr_training_pairs.append((
            string_to_ngram_ids(gt_str, ngrams),
            string_to_ngram_ids(ocr_str, ngrams)))
    return ngr_training_pairs, ngrams


def normalize_probs(probs):
    '''Normalize the probability matrix so that each row sums up to 1.'''
    row_sums = np.sum(probs, axis=1)
    weights = np.divide(np.ones(row_sums.shape), row_sums, where=row_sums > 0)
    return probs * weights[:,None] 


def initialize_probs(size, identity_weight=1, misc_weight=0.01):
    return normalize_probs(np.ones((size, size)))
#     return normalize_probs(
#         np.eye(size) * identity_weight + \
#         np.ones((size, size)) * misc_weight)


def forward(input_seq, output_seq, probs, ngr_probs):
    '''Compute the forward matrix (alpha) for the given pair
       of sequences.'''
    result = np.zeros((input_seq.shape[0]+1, output_seq.shape[0]+1))
    result[0, 0] = 1
    for i in range(1, input_seq.shape[0]+1):
        for j in range(1, output_seq.shape[0]+1):
            for k in range(min(i, input_seq.shape[1])):
                for m in range(min(j, output_seq.shape[1])):
                    x, y = input_seq[i-k-1,k], output_seq[j-m-1,m]
                    if x > -1 and y > -1:
                        result[i,j] += ngr_probs[k] * result[i-k-1,j-m-1] * probs[x,y]
    return result


def backward(input_seq, output_seq, probs, ngr_probs):
    '''Compute the backward matrix (beta) for the given pair
       of sequences.'''
    result = np.zeros((input_seq.shape[0]+1, output_seq.shape[0]+1))
    result[input_seq.shape[0], output_seq.shape[0]] = 1
    for i in range(input_seq.shape[0]-1, -1, -1):
        for j in range(output_seq.shape[0]-1, -1, -1):
            for k in range(min(input_seq.shape[0]-i, input_seq.shape[1])):
                for m in range(min(output_seq.shape[0]-j, output_seq.shape[1])):
                    x, y = input_seq[i,k], output_seq[j,m]
                    if x > -1 and y > -1:
                        result[i,j] += ngr_probs[k] * probs[x,y] * result[i+k+1,j+m+1]
    return result


def compute_expected_counts(seq_pairs, probs, ngr_probs):
    counts = np.zeros(probs.shape)
    ngr_counts = np.zeros(ngr_probs.shape)
    for input_seq, output_seq in tqdm.tqdm(seq_pairs):
        alpha = forward(input_seq, output_seq, probs, ngr_probs)
        beta = backward(input_seq, output_seq, probs, ngr_probs)
        Z = alpha[input_seq.shape[0],output_seq.shape[0]]
        for i in range(1, input_seq.shape[0]+1):
            for j in range(1, output_seq.shape[0]+1):
                if alpha[i,j]*beta[i,j] == 0:
                    continue
                co = np.zeros((min(i, input_seq.shape[1]),
                               min(j, output_seq.shape[1])))
                for k in range(min(i, input_seq.shape[1])):
                    for m in range(min(j, output_seq.shape[1])):
                        x, y = input_seq[i-k-1,k], output_seq[j-m-1,m]
                        if x > -1 and y > -1:
                            c = alpha[i-k-1,j-m-1] * ngr_probs[k] * \
                                probs[x,y] * beta[i,j] / Z
                            co[k,m] += c
                            ngr_counts[k] += c
                            counts[x,y] += c
    return counts, ngr_counts


def mean_kl_divergence(old, new):
    log_old = np.log(old, where=old > 0)
    log_new = np.log(new, where=new > 0)
    return np.sum(new*log_new - new*log_old) / new.shape[0]


def compute_new_probs(counts, probs):
    result = np.copy(probs)
    row_sums = np.sum(counts, axis=1)
    for i in range(counts.shape[0]):
        if row_sums[i] > 0:
            result[i,:] = counts[i,:] / row_sums[i]
    return result


def fit(seq_pairs, ngrams, threshold=0.0001):
    probs = initialize_probs(len(ngrams))
    ngr_probs = np.ones(3) / 3
    kl_div = np.inf
    while kl_div > threshold:
        counts, ngr_counts = compute_expected_counts(seq_pairs, probs, ngr_probs)
        new_probs = compute_new_probs(counts, probs)
        ngr_probs = ngr_counts / np.sum(ngr_counts)
        kl_div = mean_kl_divergence(probs, new_probs)
        probs = new_probs
        if np.any(probs > 1):
            raise RuntimeError('!')
        print('KL-DIV={}'.format(kl_div))
        print(ngr_probs)
    print(ngr_probs)
    return probs, ngr_probs


def matrix_to_mappings(probs, ngrams, weight_threshold=5.0):
    weights = -np.log(probs, where=(probs > 0))
    results = []
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            if probs[i,j] > 0 and weights[i,j] < weight_threshold:
                results.append((ngrams[i], ngrams[j], float(weights[i,j])))
    return results


def compile_transducer(mappings, ngr_probs, max_errors=3, max_context=3,
                       weight_threshold=5.0):
    ngr_weights = -np.log(ngr_probs)
    identity_trs, error_trs = {}, {}
    identity_mappings, error_mappings = {}, {}
    for i in range(max_context):
        identity_trs[i], error_trs[i] = [], []
        identity_mappings[i], error_mappings[i] = [], []
    for x, y, weight in mappings:
        mapping = (escape_for_pynini(x), escape_for_pynini(y), str(weight))
        if x == y:
            identity_mappings[len(x)-1].append(mapping)
        else:
            error_mappings[len(x)-1].append(mapping)
    for i in range(max_context):
        identity_trs[i] = pynini.string_map(identity_mappings[i])
        error_trs[i] = pynini.string_map(error_mappings[i])
    # TODO refactor as a subfunction
    # - build the "master transducer" containing ID-n and ERR-n symbols
    #   on transitions for n in 1..max_context and containing ngr_weights[n] in
    #   arcs leading to those
    state_dict = {}
    root = pynini.Fst()

    # FIXME refactor the merging of symbol tables into a separate function
    symbol_table = pynini.SymbolTable()
    for i in range(max_context):
        symbol_table = pynini.merge_symbol_table(symbol_table, identity_trs[i].input_symbols())
        symbol_table = pynini.merge_symbol_table(symbol_table, error_trs[i].input_symbols())
        symbol_table = pynini.merge_symbol_table(symbol_table, identity_trs[i].output_symbols())
        symbol_table = pynini.merge_symbol_table(symbol_table, error_trs[i].output_symbols())
        sym = symbol_table.add_symbol('id-{}'.format(i+1))
        sym = symbol_table.add_symbol('err-{}'.format(i+1))

    root.set_input_symbols(symbol_table)
    root.set_output_symbols(symbol_table)

    for i in range(max_errors+1):
        for j in range(max_context+1):
            s = root.add_state()
            state_dict[(i, j)] = s
            if j > 0:
                # (i, 0) -> (i, j) with epsilon
                root.add_arc(
                    state_dict[(i, 0)],
                    pynini.Arc(0, 0, ngr_weights[j-1], s))
                # (i, j) -> (i, 0) with identity
                sym = root.output_symbols().find('id-{}'.format(j))
                root.add_arc(
                    s,
                    pynini.Arc(0, sym, 0, state_dict[(i, 0)]))
                if i > 0:
                    # arc: (i-1, j) -> (i, 0) with error
                    sym = root.output_symbols().find('err-{}'.format(j))
                    root.add_arc(
                        state_dict[(i-1, j)],
                        pynini.Arc(0, sym, 0, state_dict[(i, 0)]))
        root.set_final(state_dict[(i, 0)], 0)

    root.set_start(state_dict[(0, 0)])
    replacements = []
    for i in range(max_context):
        replacements.append(('id-{}'.format(i+1), identity_trs[i]))
        replacements.append(('err-{}'.format(i+1), error_trs[i]))
    result = pynini.replace(root, replacements)
    result.invert()
    result.optimize()
    return result


def load_ngrams(filename):
    result = []
    with open(filename) as fp:
        for line in fp:
            result.append(line.replace('\n', '')) #.replace(' ', '~'))
    return result


def save_ngrams(filename, ngrams):
    with open(filename, 'w+') as fp:
        for ngr in ngrams:
            fp.write(ngr + '\n')

