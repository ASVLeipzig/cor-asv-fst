import argparse
import hfst
import math
import functools

from nltk import ngrams

# from alignment.sequence import Sequence
# import alignment
# alignment.sequence.GAP_ELEMENT = 0
# from alignment.sequence import GAP_ELEMENT
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
import difflib
GAP_ELEMENT = 0

import helper


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
                
                raw_seq = list(map(lambda x: x[0], alignment))
                gt_seq = list(map(lambda x: x[1], alignment))
                
                for n in range(1,max_n+1): # the ngrams which are considered
                    
                    raw_grams = ngrams(raw_seq, n)
                    gt_grams = ngrams(gt_seq, n)
                    
                    for raw_gram, gt_gram in zip(raw_grams, gt_grams):
                        #print(raw_gram, gt_gram)
                        
                        confusion_dicts[n][raw_gram] = confusion_dicts[n].setdefault(raw_gram, {})
                        confusion_dicts[n][raw_gram][gt_gram] = confusion_dicts[n][raw_gram].setdefault(gt_gram, 0) + 1
    
    #for i in [1, 2, 3]:
    #    print(confusion_dicts[i].items())
    
    return(confusion_dicts)


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

    items = confusion_dict.items()

    # count number of all occurrences
    s = [list(item[1].items()) for item in items]
    s = functools.reduce(lambda x, y: x + y, s, [])
    s = sum([x[1] for x in s])
    #print(s)

    # count number of ε substitutions
    epsilon_occ = sum([x[1] for x in confusion_dict.setdefault(GAP_ELEMENT, {}).items()])
    #epsilon_occ = sum([x[1] for x in confusion_dict[alignment.sequence.GAP_ELEMENT].items()])
    #print(epsilon_occ)

    # set ε to ε transitions to number of all occurrences minus ε substitutions
    # (because it models an occurrence of an ε that is not confused with an
    # existing character; this is needed for correctly calculating the
    # frequencies of ε to something transitions;
    # in the resulting (not complete) error transducer, these ε to ε transitions
    # are not preserved, but only transitions changing the input)
    if epsilon_occ != 0:
        confusion_dict[GAP_ELEMENT][GAP_ELEMENT] = s - epsilon_occ

    for item in items:
        #print(item)

        raw_gram = item[0]
        substitutions = item[1].items()

        raw_gram_freq = sum([x[1] for x in substitutions])
        #print(raw_gram_freq)

        for sub in substitutions:
            frequency_list.append((raw_gram, sub[0], sub[1] / raw_gram_freq))

    #print(sorted(frequency_list, key=lambda x: x[2]))
    return frequency_list


def write_frequency_list(frequency_list, filename):
    """Write human-readable (as string) frequency_list to filename (tab-separated)."""

    with open(filename, 'w') as f:
        for raw_gram, gt_gram, freq in frequency_list:
            f.write(u''.join(map(lambda x: x if x != GAP_ELEMENT else u'□', raw_gram)) + u'\t' +
                    u''.join(map(lambda x: x if x != GAP_ELEMENT else u'□', gt_gram)) + u'\t' +
                    str(freq) + u'\n')
    return


# def read_frequency_list(filename):
#     """Read frequency_list from filename."""

#     freq_list = []
#     with open(filename, 'r') as f:
#         for line in f:
#             instr, outstr, freq = line.strip('\n').split('\t')
#             freq_list.append((instr, outstr, float(freq))) # will not work with lists
#     return freq_list


def transducer_from_list(confusion_list, frequency_class=False, identity_transitions=False):
    """
    Convert a list of tuples: input_gram, output_gram, relative_frequency,
    into a weighted transducer performing the given transductions with 
    the encountered probabilities. 
    If frequency_class is True, then assume frequency classes/ranks are given
    instead of the relative frequencies, so no logarithm of the values
    will be performed for obtaining the weight.
    """
    
    if frequency_class:
        #confusion_list_log = list(map(lambda x: (x[0], x[1], x[2]), confusion_list))
        confusion_list_log = confusion_list
    else:
        confusion_list_log = list(map(lambda x: (x[0], x[1], -math.log(x[2])), confusion_list))

    fst_dict = {}

    for in_gram, out_gram, weight in confusion_list_log:
        
        # filter out epsilon -> epsilon transition
        if (in_gram.count(GAP_ELEMENT) < len(in_gram) or
            out_gram.count(GAP_ELEMENT) < len(out_gram)):
            # filter out identity transitions if identity_transitions=False (useful for max_errors per window)
            if identity_transitions or in_gram != out_gram:
                regap = lambda x: x if x != GAP_ELEMENT else hfst.EPSILON
                instr = ''.join(map(regap, in_gram))
                outstr = ''.join(map(regap, out_gram))
                fst_dict[instr] = fst_dict.setdefault(instr, []) + [(outstr, weight)]

    confusion_fst = hfst.fst(fst_dict)
    #confusion_fst.substitute(GAP_ELEMENT, hfst.EPSILON, input=bool, output=bool)

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
    as well as punctuation-only edits.
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
    if gt_char == "ͤ" or gt_char != GAP_ELEMENT and gt_char.isalnum():
        return False

    # alphanumeric to epsilon or space
    if gt_char in [GAP_ELEMENT, " "] and (raw_char == "ͤ" or raw_char != GAP_ELEMENT and raw_char.isalnum()):
        return False

    # all other edits modify output punctuation
    return True


def no_punctuation_edits(confusion):
    """
    Take one confusion entry and return True iff 
    none of the n-gram positions contain edits that 
    would convert some character into punctuation.
    """
    
    for in_char, out_char in zip(confusion[0],confusion[1]):
        if is_punctuation_edit(in_char, out_char):
            return False
    return True

def main():
    """
    Read GT and OCR files following the path scheme <directory>/<ID>.<suffix>
    and <directory>/<ID>.gt.txt, where each file contains one line of text.
    Align GT and OCR lines for same IDs, count n-gram pair occurrences of n={1,2,3}
    for differing lines.
    Write confusions frequencies to confusion_<n>.txt, 
    and create a simple error_transducer_<n>.hfst.
    """

    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst error model creator')
    parser.add_argument('directory', metavar='PATH', help='directory for input and GT files')
    parser.add_argument('-I', '--input-suffix', metavar='SUF', type=str, default='txt', help='input (OCR) filenames suffix')
    parser.add_argument('-C', '--max-context', metavar='NUM', type=int, default=3, help='maximum size of context count edits at')
    parser.add_argument('-P', '--preserve-punctuation', action='store_true', default=False, help='ignore edits to/from non-alphanumeric or non-space characters')
    args = parser.parse_args()
    
    # read GT data and OCR data (from dta19_reduced)
    #path = '../dta19-reduced/traindata/'
    gt_dict = helper.create_dict(args.directory + "/", 'gt.txt')

    #frak3_dict = create_dict(path, 'deu-frak3')
    #fraktur4_dict = helper.create_dict(path, 'Fraktur4')
    #foo4_dict = create_dict(path, 'foo4')
    ocr_dict = helper.create_dict(args.directory + "/", args.input_suffix)

    # get list of confusion dicts with different context sizes
    confusion_dicts = get_confusion_dicts(gt_dict, ocr_dict, args.max_context)

    #n = 3
    #preserve_punctuation = False

    for n in range(1,args.max_context+1): # considered ngrams

        print('n: ', str(n))

        # convert to relative frequencies
        confusion_list = preprocess_confusion_dict(confusion_dicts[n])

        # write confusion to confusion_<n>.txt
        write_frequency_list(confusion_list, 'confusion_' + str(n) + '.txt')
        print('length of confusion list for context size n:', len(confusion_list))

        if args.preserve_punctuation:
            confusion_list = list(filter(no_punctuation_edits,confusion_list))

        print('length of confusion list after filtering:', len(confusion_list))

        # create (non-complete) error_transducer and optimize it
        error_transducer = transducer_from_list(confusion_list)
        optimize_error_transducer(error_transducer)

        # write transducer to file
        helper.save_transducer('error_transducer_' + str(n) + '.hfst', error_transducer)
        #error_transducer = helper.load_transducer('error_transducer_' + str(n) + '.hfst')

        #print(error_transducer)

if __name__ == '__main__':
    main()
