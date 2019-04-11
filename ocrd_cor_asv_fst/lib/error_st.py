# FIXME currently deprecated -- fix to use pynini instead of hfst

'''Error model based on a stochastic transducer.'''
# TODO literature, computing P(y|x) rather than P(x, y) etc.

import argparse
from collections import defaultdict
import hfst
import numpy as np
from operator import itemgetter
import tqdm

import sys
import helper


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
        raise Exception('Number of unigrams exceeds the number of allowed'
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
    # print(string, result)
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
        # if Z <= 0:
        #     print('Skipping')
        #     print(alpha)
        #     continue
        # if Z > 1:
        #     raise Exception('Z > 1')
#         print(Z)
#         print(alpha[input_seq.shape[0],output_seq.shape[0]])
#         print(beta[0,0])
#         print(alpha)
#         print(beta)
#         print(input_seq, output_seq)
        for i in range(1, input_seq.shape[0]+1):
            for j in range(1, output_seq.shape[0]+1):
                if alpha[i,j]*beta[i,j] == 0:
                    continue
                # Z = alpha[i,j]*beta[i,j]
                co = np.zeros((min(i, input_seq.shape[1]), min(j, output_seq.shape[1])))
                for k in range(min(i, input_seq.shape[1])):
                    for m in range(min(j, output_seq.shape[1])):
                        x, y = input_seq[i-k-1,k], output_seq[j-m-1,m]
                        if x > -1 and y > -1:
#                             print(x, y, alpha[i-k-1,j-m-1], probs[x,y], beta[i,j], Z,
#                                 alpha[i-k-1,j-m-1] * probs[x,y] * beta[i,j] / Z)
                            c = alpha[i-k-1,j-m-1] * ngr_probs[k] * \
                                probs[x,y] * beta[i,j] / Z
                            co[k,m] += c
                            ngr_counts[k] += c
                            counts[x,y] += c
                # TODO alpha[i,j]*beta[i,j] should equal to np.sum(co) everywhere (?)
                # print(i, j, beta[i,j], Z, np.sum(co), alpha[i,j]*beta[i,j]/Z)
                # print(i, j, beta[i,j], Z, np.sum(co))
                # print(co)
        # raise NotImplementedError()
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
    # ngr_probs = np.ones(3)
    kl_div = np.inf
    while kl_div > threshold:
        counts, ngr_counts = compute_expected_counts(seq_pairs, probs, ngr_probs)
        new_probs = compute_new_probs(counts, probs)
        ngr_probs = ngr_counts / np.sum(ngr_counts)
#         new_probs = counts / np.sum(counts)
        kl_div = mean_kl_divergence(probs, new_probs)
        probs = new_probs
        if np.any(probs > 1):
            raise RuntimeError('!')
        print('KL-DIV={}'.format(kl_div))
        print(ngr_probs)
#         mappings = matrix_to_mappings(probs, ngrams, weight_threshold=5)
#         for (input_str, output_str, weight) in mappings:
#             print('\'{}\' : \'{}\' :: {}'.format(input_str, output_str, np.exp(-weight)))
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


def align_mappings(mappings):

    def _align(x, y):
        return tuple(zip(tuple(x) + (hfst.EPSILON,)*max(0, len(y)-len(x)),
                         tuple(y) + (hfst.EPSILON,)*max(0, len(x)-len(y))))
    
    # convert tuples like ('abc', 'def', 1.3)
    # to ((('a', 'd'), ('b', 'e'), ('c', 'f')), 3, 1.3)
    # (the number "3" here is len(x) -- the length of the used context)
    # insert epsilons where necessary
    result = []
    for x, y, weight in mappings:
        result.append((_align(x, y), len(x), weight))
    return result


def compile_transducer(mappings, ngr_probs, max_errors=3, max_context=3,
                       weight_threshold=5.0):
    '''Convert the trained probability matrix to a transducer.'''
    ngr_weights = -np.log(ngr_probs)
    print(ngr_weights)

    class AlignmentTrie:
        def __init__(self):
            self.children = {}
            self.final_weight = None
            self.is_identity = True

        def insert(self, alignment, weight):
            if not alignment:
                self.final_weight = weight
            else:
                x, y = alignment[0]
                if (x, y) not in self.children:
                    self.children[(x, y)] = AlignmentTrie()
                    self.children[(x, y)].is_identity = \
                        self.is_identity and x == y
                self.children[(x, y)].insert(alignment[1:], weight)

    def _convert_tries_to_single_error_tr(tries, only_identity=False):

        def _process_node(tr_b, node, state):
            if node.final_weight is not None:
                if not only_identity or node.is_identity:
                    target_state = 0 if node.is_identity else 1
                    tr_b.add_transition(
                        state,
                        hfst.HfstBasicTransition(
                            target_state,
                            hfst.EPSILON,
                            hfst.EPSILON,
                            node.final_weight))
            for (x, y), child in node.children.items():
                target_state = tr_b.add_state()
                # print(x, y, target_state)
                tr_b.add_transition(
                    state,
                    hfst.HfstBasicTransition(target_state, x, y, 0.0))
                _process_node(tr_b, child, target_state)

        tr_b = hfst.HfstBasicTransducer()
        tr_b.set_final_weight(0, 0.0)
        if not only_identity:
            tr_b.add_state()        # 1 -- ending state with one error
            tr_b.set_final_weight(1, 0.0)
        for i in range(max_errors):
            state = tr_b.add_state()
            tr_b.add_transition(
                0,
                hfst.HfstBasicTransition(
                    state,
                    hfst.EPSILON,
                    hfst.EPSILON,
                    ngr_weights[i]))
            _process_node(tr_b, tries[i], state)
        tr = hfst.HfstTransducer(tr_b)
        print('Minimizing...')
        # tr.minimize()
        return tr

    # states are identified by tuples: (e, n, m), where:
    # - e -- number of errors until now
    # - n -- the length of context chosen for next transition (or 0 if not yet
    #        chosen)
    # - m -- the length of the already processed context
    tr_b = hfst.HfstBasicTransducer()
    alignment_tries = [AlignmentTrie() for i in range(max_context)]
    aligned_mappings = align_mappings(mappings)
    # add flags -- FIXME this is a quick-and-dirty solution!!!
    aligned_mappings.extend(
        [((('@N.{}@'.format(chr(c)), '@N.{}@'.format(chr(c))),), 1, 0.0) \
         for c in range(ord('A'), ord('Z')+1)])
    for alignment, context_len, weight in aligned_mappings:
        if context_len <= max_context:
            alignment_tries[context_len-1].insert(alignment, weight)
    tr = hfst.epsilon_fst()
    for i in range(max_errors):
        tr.concatenate(_convert_tries_to_single_error_tr(alignment_tries))
    tr.concatenate(_convert_tries_to_single_error_tr(alignment_tries, only_identity=True))
    # TODO multiple errors
    # raise NotImplementedError()
    # tr.remove_epsilons()
    # tr.push_weights_to_start()
    tr.invert()
    return tr


def compile_error_transducer(ocr_dict, gt_dict, max_context=3, max_errors=3):
    raise NotImplementedError()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction ocrd-cor-asv-fst ST error model '
                    'creator')
    parser.add_argument(
        'directory', metavar='PATH',
        help='directory (or CSV file) for input and GT files')
    parser.add_argument(
        '-I', '--input-suffix', metavar='SUF', type=str, default='txt',
        help='input (OCR) filenames suffix')
    parser.add_argument(
        '-G', '--gt-suffix', metavar='SUF', type=str, default='gt.txt',
        help='clean (Ground Truth) filenames suffix')
    parser.add_argument(
        '-C', '--max-context', metavar='NUM', type=int, default=3,
        help='maximum size of context count edits at')
    parser.add_argument(
        '-E', '--max-errors', metavar='NUM', type=int, default=3,
        help='maximum number of errors the resulting FST can correct '
             '(applicable within one window, i.e. a certain number of words)')
    parser.add_argument(
        '-o', '--output-file', metavar='FILE', type=str,
        default='error.fst', help='file to store the resulting transducer')
    parser.add_argument(
        '-w', '--weights-file', metavar='FILE', type=str,
        help='file to store the trained weights')
    parser.add_argument(
        '-W', '--load-weights-from', metavar='FILE', type=str,
        help='load weights from FILE instead of training')
    parser.add_argument(
        '-n', '--max-ngrams', metavar='NUM', type=str, default=1000,
        help='maximum number of n-grams')
    parser.add_argument(
        '-N', '--ngrams-file', metavar='FILE', type=str, default='ngrams.txt',
        help='file to save/load n-grams to/from')
    parser.add_argument(
        '-t', '--weight-threshold', metavar='VAL', type=float, default=5.0,
        help='discard transitions with weight higher than VAL')
    return parser.parse_args()


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


def main():
    args = parse_arguments()

    # if weight file given -> load weights from there, otherwise train them
    ngrams, probs, ngr_probs = None, None, None
    if args.load_weights_from is not None:
        ngrams = load_ngrams(args.ngrams_file)
        with np.load(args.load_weights_from) as data:
            probs, ngr_probs = data['probs'], data['ngr_probs']
    else:
        ocr_dict = helper.create_dict(args.directory, args.input_suffix)
        gt_dict = helper.create_dict(args.directory, args.gt_suffix)
        training_pairs, ngrams = preprocess_training_data(
            ocr_dict, gt_dict,
            max_n=args.max_context, max_ngrams=1000)
        save_ngrams(args.ngrams_file, ngrams)
        probs, ngr_probs = fit(training_pairs, ngrams, threshold=0.001)
        if args.weights_file is not None:
            np.savez(args.weights_file, probs=probs, ngr_probs=ngr_probs)

    mappings = matrix_to_mappings(
        probs, ngrams, weight_threshold=args.weight_threshold)
    for input_str, output_str, weight in mappings:
        print('\''+input_str+'\'', '\''+output_str+'\'', weight, sep='\t')
    # for alignment, context_len, weight in align_mappings(mappings):
    #     print(alignment, context_len, weight)
    tr = compile_transducer(
        mappings, ngr_probs, max_errors=args.max_errors,
        max_context=args.max_context, weight_threshold=args.weight_threshold)
    tr.write_to_file(args.output_file)


if __name__ == '__main__':
    main()

