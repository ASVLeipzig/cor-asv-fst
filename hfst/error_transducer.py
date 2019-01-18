"""
Create error correcting string transducers trained 
from paired OCR / ground truth text data.
"""
from io import open
import os
import argparse
import math
import csv

import hfst
from nltk import ngrams

# from alignment.sequence import Sequence
# import alignment
# alignment.sequence.GAP_ELEMENT = ' '
# from alignment.sequence import GAP_ELEMENT
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
import difflib
# gap/epsilon needs to be a character so we can easily make a transducer from it,
#             but it must not ever occur in input
GAP_ELEMENT = u' ' # (nbsp) # '\0' # (nul breaks things in libhfst)

import helper
from sliding_window import FlagEncoder

def get_confusion_dicts(gt_dict, raw_dict, max_n):
    """
    Take two dictionaries, mapping: id -> line for GT and OCR, 
    and align corresponding lines. 
    If they are different, split into n-grams for n={1,2,3},
    and count n-gram occurances.
    Return these counts as a list of dicts:
    [ignored, 1-grams, 2-grams, 3-grams]
    """

    corresponding_list = []     # list of tuples (gt_line, raw_line)
    difference_list = []        # list of tuples (gt_line, raw_line) with gt_line != raw_line

    # divide sentences in those containing OCR errors and the rest
    for key in gt_dict.keys():
        raw_line = raw_dict.get(key, None)
        gt_line = gt_dict[key]
        if raw_line != None:
            corresponding = (gt_line, raw_line)
            corresponding_list.append(corresponding)
            if raw_line != gt_line:
                difference_list.append(corresponding)

    # each dict in this list contains counts of character ngram confusions up to
    # the position in the list (thus, up to 3grams are considered);
    # position 0 is ignored:
    # [ignored, 1grams, 2grams, 3grams]
    confusion_dicts = [{}, {}, {}, {}]
    
    matcher = difflib.SequenceMatcher(isjunk=None, autojunk=False) # disable "junk" detection heuristics (mainly for source code)
    
    for gt_line, raw_line in corresponding_list: # difference_list
    #for (gt_line, raw_line) in difference_list[1:100]:

        #print(gt_line)
        #print(raw_line)
        if not gt_line or not raw_line:
            continue
        if GAP_ELEMENT in gt_line or GAP_ELEMENT in raw_line:
            raise Exception('gap element must not occur in text', GAP_ELEMENT, raw_line, gt_line)
        
        # alignment of lines

        # a = Sequence(raw_line)
        # b = Sequence(gt_line)
        # # create a vocabulary and encode the sequences
        # v = Vocabulary()
        # aEncoded = v.encodeSequence(a)
        # bEncoded = v.encodeSequence(b)
        # # create a scoring and align the sequences using global aligner
        # scoring = SimpleScoring(2, -1)
        # aligner = StrictGlobalSequenceAligner(scoring, -2)
        # score = aligner.align(aEncoded, bEncoded)
        # if score < -10 and score < 5-len(gt_line):
        #     #print('ignoring bad OCR:')
        #     #print(raw_line)
        #     #print(gt_line)
        #     continue
        # score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)
        # if encodeds:
        #     alignment = v.decodeSequenceAlignment(encodeds[0]) # take only 1-best alignment (others are suboptimal)
        #     #print(alignment)
        #     #print('Alignment score:', alignment.score)
        #     #print('Percent identity:', alignment.percentIdentity())

        #     if alignment.percentIdentity() > 5: #alignment.percentIdentity() < 100:

        matcher.set_seqs(raw_line, gt_line)
        if matcher.quick_ratio() < 0.1 and len(gt_line) > 5:
            #print('ignoring bad OCR:')            
            #print(raw_line)
            #print(gt_line)
            continue
        else:
            alignment = []
            for op, raw_begin, raw_end, gt_begin, gt_end in matcher.get_opcodes():
                if op == 'equal':
                    alignment.extend(zip(raw_line[raw_begin:raw_end], gt_line[gt_begin:gt_end]))
                elif op == 'replace': # not really substitution:
                    delta = raw_end-raw_begin-gt_end+gt_begin
                    #alignment.extend(zip(raw_line[raw_begin:raw_end] + [GAP_ELEMENT]*(-delta), gt_line[gt_begin:gt_end] + [GAP_ELEMENT]*(delta)))
                    if delta > 0: # replace+delete
                        alignment.extend(zip(raw_line[raw_begin:raw_end-delta], gt_line[gt_begin:gt_end]))
                        alignment.extend(zip(raw_line[raw_end-delta:raw_end], [GAP_ELEMENT]*(delta)))
                    if delta <= 0: # replace+insert
                        alignment.extend(zip(raw_line[raw_begin:raw_end], gt_line[gt_begin:gt_end+delta]))
                        alignment.extend(zip([GAP_ELEMENT]*(-delta), gt_line[gt_end+delta:gt_end]))
                elif op == 'insert':
                    alignment.extend(zip([GAP_ELEMENT]*(gt_end-gt_begin), gt_line[gt_begin:gt_end]))
                elif op == 'delete':
                    alignment.extend(zip(raw_line[raw_begin:raw_end], [GAP_ELEMENT]*(raw_end-raw_begin)))
                else:
                    raise Exception("difflib returned invalid opcode", op, "in", gt_line)
            assert raw_end == len(raw_line)
            assert gt_end == len(gt_line)
            
            if alignment:
                
                raw_aligned = ''.join(map(lambda x: x[0], alignment))
                gt_aligned = ''.join(map(lambda x: x[1], alignment))
                
                for n in range(1, max_n+1): # the ngrams which are considered
                    
                    raw_ngrams = ngrams(raw_aligned, n)
                    gt_ngrams = ngrams(gt_aligned, n)
                    
                    for raw_ngram, gt_ngram in zip(raw_ngrams, gt_ngrams):
                        #print(raw_ngram, gt_ngram)
                        raw_string = ''.join(raw_ngram)
                        gt_string = ''.join(gt_ngram)
                        
                        confusion_dicts[n][raw_string] = confusion_dicts[n].setdefault(raw_string, {})
                        confusion_dicts[n][raw_string][gt_string] = confusion_dicts[n][raw_string].setdefault(gt_string, 0) + 1
    
    #for i in [1, 2, 3]:
    #    print(confusion_dicts[i].items())
    
    return confusion_dicts


def preprocess_confusion_dict(confusion_dict):
    """
    Convert confusion dictionary (for one n),
    mapping: input_ngram, output_ngram -> count,
    to a list with relative frequencies 
    (in relation to the total number of that input ngram, 
    not of all input ngrams).
    Return a list of tuples:
    input_ngram, output_ngram, frequency.
    """
    
    #Convert list of form ((input_string, output_string),
    #count) into list of form (((input_string, output_string),
    #relative_frequency), excluding infrequent errors,
    #maybe smoothing (not implemented)."""

    frequency_list = []

    raw_items = confusion_dict.items()

    # count number of all occurrences
    total_freq = sum([sum(freq
                          for gt_ngram, freq in gt_dict.items())
                      for raw_ngram, gt_dict in raw_items])
    print('total edit count:', total_freq)

    # count number of ε-substitutions
    epsilon_freq = sum([gap_freq
                        for gap_ngram, gap_freq in confusion_dict.setdefault(GAP_ELEMENT, {}).items()])
    #epsilon_freq = sum([gap_freq for gap_ngram, gap_freq in confusion_dict[GAP_ELEMENT].items()])
    print('insertion count:', epsilon_freq)

    # set ε-to-ε transitions to number of all occurrences minus ε-substitutions
    # (because it models an occurrence of an ε that is not confused with an
    # existing character; this is needed for correctly calculating the
    # frequencies of ε to something transitions;
    # in the resulting (not complete) error transducer, these ε to ε transitions
    # are not preserved, but only transitions changing the input)
    if epsilon_freq != 0:
        confusion_dict[GAP_ELEMENT][GAP_ELEMENT] = total_freq - epsilon_freq

    for raw_ngram, gt_dict in raw_items:
        substitutions = gt_dict.items()
        total_freq = sum([freq for gt_ngram, freq in substitutions])

        for gt_ngram, freq in substitutions:
            frequency_list.append((raw_ngram, gt_ngram, freq / total_freq))

    #print(sorted(frequency_list, key=lambda x: x[2]))
    return frequency_list


def write_frequency_list(frequency_list, filename):
    """Write human-readable (as string) frequency_list to filename (tab-separated)."""

    with open(filename, mode='w', encoding='utf-8') as f:
        for raw_gram, gt_gram, freq in frequency_list:
            f.write(raw_gram.replace(GAP_ELEMENT, u'□') + u'\t' +
                    gt_gram.replace(GAP_ELEMENT, u'□') + u'\t' +
                    str(freq) + u'\n')
    return


def read_frequency_list(filename):
    """Read frequency_list from filename."""

    freq_list = []
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            instr, outstr, freq = line.strip('\n').split('\t')
            freq_list.append((instr.replace(u'□', GAP_ELEMENT),
                              outstr.replace(u'□', GAP_ELEMENT),
                              float(freq)))
    return freq_list


def transducer_from_list(confusion_list, frequency_class=False, weight_threshold=7.0, identity_transitions=False):
    """
    Convert a list of tuples: input_gram, output_gram, relative_frequency,
    into a weighted transducer performing the given transductions with 
    the encountered probabilities. 
    If frequency_class is True, then assume frequency classes/ranks are given
    instead of the relative frequencies, so no logarithm of the values
    will be performed for obtaining the weight.
    If identity_transitions is True, then keep non-edit transitions.
    If weight_threshold is given, then prune away those transitions with 
    a weight higher than that.
    """
    
    if frequency_class:
        #confusion_list_log = list(map(lambda x: (x[0], x[1], x[2]), confusion_list))
        confusion_list_log = confusion_list
    else:
        confusion_list_log = list(map(lambda x: (x[0], x[1], -math.log(x[2])), confusion_list))

    confusion_fst = hfst.HfstBasicTransducer()
    # not as good as using .substitute() afterwards: tokenizer based...
    # tok = hfst.HfstTokenizer()
    # tok.add_skip_symbol(GAP_ELEMENT)
    # not good at all (and slow): dictionary based...
    # fst_dict = {}

    for in_gram, out_gram, weight in confusion_list_log:

        # prune away rare edits:
        if weight_threshold and weight > weight_threshold:
            continue
        # filter out identity transitions (unless identity_transitions=True):
        if in_gram == out_gram and not identity_transitions:
            continue
        # instr = in_gram.replace(GAP_ELEMENT, "")
        # outstr = out_gram.replace(GAP_ELEMENT, "")
        # # filter out ε-to-ε transitions:
        # if not instr and not outstr:
        #     continue
        # # avoid hfst error 'Empty word.':
        # if not instr:
        #     instr = hfst.EPSILON
        # if not outstr:
        #     outstr = hfst.EPSILON
        # fst_dict[instr] = fst_dict.setdefault(instr, []) + [(outstr, weight)]
        # much faster than hfst.fst(dict):
        #confusion_fst.disjunct(tok.tokenize(in_gram, out_gram), weight)
        # tokenize makes suboptimal aligments:
        confusion_fst.disjunct(tuple(zip(in_gram, out_gram)), weight)
    
    #print('creating confusion fst for %d input n-grams' % len(fst_dict))
    #confusion_fst = hfst.fst(fst_dict) # does not respect epsilon/gap and multi-character symbols (for flags at runtime)
    # maybe we can keep this approach with hfst.tokenized_fst()?
    confusion_fst.substitute(GAP_ELEMENT, hfst.EPSILON, input=True, output=True)

    # make sure the error transducer already contains the flag symbols
    # needed for the sliding window construction (to avoid the need to
    # merge symbol tables on the input transducer during composition):
    confusion_fst = hfst.HfstTransducer(confusion_fst)
    flag_encoder = FlagEncoder()
    for flag in flag_encoder.flag_list:
        confusion_fst.insert_to_alphabet(flag)
    return confusion_fst


def optimize_error_transducer(error_transducer):
    """Optimize error_transducer by minimizing, removing epsilon und
    pushing weights to start."""

    # TODO: Weights should not be at the start, but at the actual character
    # that is modified to ensure that the weights are local.

    error_transducer.minimize()
    error_transducer.remove_epsilons()
    error_transducer.push_weights_to_start()


def is_punctuation_edit(raw_char, gt_char):
    """
    Return True iff an edit of raw_char to gt_char could be
    a punctuation edit.
    Punctuation characters only matter if on the GT side, i.e.
    allow edits from punctuation to alphanumeric characters,
    (because those often occur inside words), but forbid 
    edits from alphanumeric to punctuation characters,
    as well as punctuation-only edits (because those likely
    cannot be corrected with a lexicon).
    Whether or not that edit is part of a punctuation edit
    still depends on the context, though.
    """

    # no edit
    if raw_char == gt_char:
        return False

    # segmentation error
    if raw_char in [GAP_ELEMENT, " "] and gt_char in [GAP_ELEMENT, " "]:
        return False

    # edit to an alphanumeric character
    if gt_char == "\u0364" or gt_char != GAP_ELEMENT and gt_char.isalnum():
        return False

    # alphanumeric to epsilon or space
    if gt_char in [GAP_ELEMENT, " "] and (raw_char == "\u0364" or raw_char != GAP_ELEMENT and raw_char.isalnum()):
        return False

    # all other edits modify output punctuation
    return True


def no_punctuation_edits(confusion):
    """
    Take one confusion entry and return True iff 
    none of the n-gram positions contain edits that 
    would convert some character into punctuation.
    """
    
    for in_char, out_char in zip(confusion[0], confusion[1]):
        if is_punctuation_edit(in_char, out_char):
            return False
    return True

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction ocrd-cor-asv-fst error model creator')
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
        '-P', '--preserve-punctuation', action='store_true', default=False,
        help='ignore edits to/from non-alphanumeric or non-space characters')
    return parser.parse_args()

def main():
    """
    Read GT and OCR files following the path scheme
    <directory>/<ID>.<input_suffix> and <directory>/<ID>.<gt_suffix>, where
    each file contains one line of text. Align GT and OCR lines for same IDs,
    count n-gram pair occurrences of n={1,2,3} for differing lines.
    Write confusions frequencies to confusion_<n>.txt, 
    and create a simple error_transducer_<n>.hfst.
    """

    args = parse_arguments()

    if os.path.isdir(args.directory):
        if os.access(args.directory, os.R_OK|os.X_OK):
            gt_dict = helper.create_dict(args.directory, args.gt_suffix)
            ocr_dict = helper.create_dict(args.directory, args.input_suffix)
        else:
            raise argparse.ArgumentTypeError("not allowed to read directory %s" % args.directory)
    elif os.path.isfile(args.directory):
        if os.access(args.directory, os.R_OK):
            class gt_table(csv.Dialect):
                delimiter = '\t'
                quotechar = None
                escapechar = None
                doublequote = False
                skipinitialspace = False
                lineterminator = '\n'
                quoting = csv.QUOTE_NONE
            
            with open(args.directory, mode='r', encoding='utf-8') as gt_file:
                gt_reader = csv.reader(gt_file, dialect=gt_table())
                gt_dict = {}
                ocr_dict = {}
                for i, (ocr, gt) in enumerate(gt_reader):
                    ocr_dict[i] = ocr
                    gt_dict[i] = gt
        else:
            raise argparse.ArgumentTypeError("not allowed to read file %s" % args.directory)
    else:
        raise argparse.ArgumentTypeError("invalid path %s" % args.directory)

    # get list of confusion dicts with different context sizes
    confusion_dicts = get_confusion_dicts(gt_dict, ocr_dict, args.max_context)

    #n = 3
    #preserve_punctuation = False

    for n in range(1, args.max_context+1): # considered ngrams

        print('n: ', str(n))

        # convert to relative frequencies
        confusion_list = preprocess_confusion_dict(confusion_dicts[n])

        # write confusion to confusion_<n>.txt
        write_frequency_list(confusion_list, 'confusion_' + str(n) + '.txt')
        print('length of confusion list for context size n:', len(confusion_list))

        if args.preserve_punctuation:
            confusion_list = list(filter(no_punctuation_edits, confusion_list))

        print('length of confusion list after filtering:', len(confusion_list))

        # create (non-complete) error_transducer and optimize it
        error_transducer = transducer_from_list(confusion_list)
        optimize_error_transducer(error_transducer)

        # write transducer to file
        error_transducer.write_to_file('error_transducer_' + str(n) + '.hfst')

if __name__ == '__main__':
    main()
