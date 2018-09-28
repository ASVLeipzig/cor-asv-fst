import hfst
import math
import functools

from nltk import ngrams

from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = "ε"
# TODO: this GAP_ELEMENT will cause problems, when greek text is processed
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

import helper


def get_confusion_dict(gt_dict, raw_dict):
    """Takes a gt_dict: id -> line and a raw_dict: id -> line and aligns
    corresponding lines. If the corresponding lines are different, they are
    split into ngrams (of lengths 1 to 3) and each occurrence of
    ngram pairs is counted (separately for each ngram length, also if the ngrams
    are identical).
    These counts are returned as a list of dicts:
    [ignored, 1grams, 2grams, 3grams]"""

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
    confusion_dict = [{}, {}, {}, {}]

    for (gt_line, raw_line) in difference_list:
    #for (gt_line, raw_line) in difference_list[1:100]:

        #print(gt_line)
        #print(raw_line)

        # alignment of lines

        a = Sequence(raw_line)
        b = Sequence(gt_line)

        # create a vocabulary and encode the sequences
        v = Vocabulary()
        aEncoded = v.encodeSequence(a)
        bEncoded = v.encodeSequence(b)

        # create a scoring and align the sequences using global aligner
        scoring = SimpleScoring(2, -1)
        aligner = GlobalSequenceAligner(scoring, -2)
        score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

        for encoded in encodeds:
            alignment = v.decodeSequenceAlignment(encoded)
            print(alignment)
            print('Alignment score:', alignment.score)
            print('Percent identity:', alignment.percentIdentity())

            if alignment.percentIdentity() < 100:

                firsts = ''.join(list(map(lambda x: x[0], alignment)))
                seconds = ''.join(list(map(lambda x: x[1], alignment)))

                for n in [1, 2, 3]: # the ngrams which are considered

                    grams_first = list(ngrams(firsts, n))
                    grams_second = list(ngrams(seconds, n))

                    for i, gram in enumerate(grams_first):
                        first = ''.join(gram)
                        second = ''.join(grams_second[i])
                        #print(first, second)

                        confusion_dict[n][first] = confusion_dict[n].setdefault(first, {})
                        confusion_dict[n][first][second] = confusion_dict[n][first].setdefault(second, 0) + 1


    #for i in [1, 2, 3]:
    #    print(confusion_dict[i].items())

    return(confusion_dict)


def preprocess_confusion_dict(confusion_dict):
    """Convert confusion_dict: input_ngram, output_ngram -> count
    to a list with relative frequencies (in relation to the summed counts
    of the input ngram, not all input ngrams)."""

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
    epsilon_occ = sum([x[1] for x in confusion_dict.setdefault('ε', {}).items()])
    #epsilon_occ = sum([x[1] for x in confusion_dict['ε'].items()])
    #print(epsilon_occ)

    # set ε to ε transitions to number of all occurrences minus ε substitutions
    # (because it models an occurrence of an ε that is not confused with an
    # existing character; this is needed for correctly calculating the
    # frequencies of ε to something transitions;
    # in the resulting (not complete) error transducer, these ε to ε transitions
    # are not preserved, but only transitions changing the input)
    if epsilon_occ != 0:
        confusion_dict['ε']['ε'] = s - epsilon_occ

    for item in items:
        #print(item)

        input_str = item[0]
        substitutions = item[1].items()

        input_str_freq = sum([x[1] for x in substitutions])
        #print(input_str_freq)

        for sub in substitutions:
            frequency_list.append((input_str, sub[0], str(sub[1] / input_str_freq)))

    #print(sorted(frequency_list, key=lambda x: x[2]))
    return frequency_list


def write_frequency_list(frequency_list, filename):
    """Write frequency_list to filename (tab-separated)."""

    with open(filename, 'w') as f:
        for entry in frequency_list:
            f.write('\t'.join(entry) + '\n')
    return


def read_frequency_list(filename):
    """Read frequency_list from filename."""

    freq_list = []
    with open(filename, 'r') as f:
        for line in f:
            instr, outstr, freq = line.strip('\n').split('\t')
            freq_list.append((instr, outstr, float(freq)))
    return freq_list


def transducer_from_list(confusion_list, frequency_class=False, identity_transitions=False):
    """Converts list of form (input_string, output_string, relative_frequency)
    into a weighted transducer performing the given transductions with the encountered
    probabilities. If frequency_class is True, instead of the relative
    frequency, a frequency class is given and no logarithm of the value
    will be performed for obtaining the weight."""

    if frequency_class:
        #confusion_list_log = list(map(lambda x: (x[0], x[1], x[2]), confusion_list))
        confusion_list_log = confusion_list
    else:
        confusion_list_log = list(map(lambda x: (x[0], x[1], -math.log(x[2])), confusion_list))

    fst_dict = {}

    for entry in confusion_list_log:
        instr, outstr, weight = entry

        # filter epsilon -> epsilon transition
        if (instr != 'ε' or outstr != 'ε'):
            #  filter identity transitions if identity_transitions=False
            if identity_transitions or instr != outstr:
                fst_dict[instr] = fst_dict.setdefault(instr, []) + [(outstr, weight)]

    confusion_fst = hfst.fst(fst_dict)
    confusion_fst.substitute('ε', hfst.EPSILON, input=bool, output=bool)

    return confusion_fst


def optimize_error_transducer(error_transducer):
    """Optimize error_transducer by minimizing, removing epsilon und
    pushing weights to start."""

    # TODO: Weights should not be at the start, but at the actual character
    # that is modified to ensure that the weights are local.

    error_transducer.minimize()
    error_transducer.remove_epsilons()
    error_transducer.push_weights_to_start()


def is_punctuation_edit(char1, char2):
    """Check if an edit of char1 to char2 is a punctuation edit.
    Edits from punctuation to alphanumerical characters are allowed,
    because they often occur inside words. Edits from alphanumeric
    characters to punctuation are not allowed."""

    # TODO: epsilon should not be treated as a normal character to prevent
    # confusion with the greek character

    # no edit
    if char1 == char2:
        return False

    # segmentation errors
    if char1 in ["ε", " "] and char2 in ["ε", " "]:
        return False

    # edit to an alphanumeric character
    if (char2.isalnum() or char2 == "'ͤ") and not char2 in ["ε", " "]:
        return False

    # alphanumeric to epsilon or space
    if (char1.isalnum() or char1 == "'ͤ") and char2 in ["ε", " "]:
        return False

    # all other edits modify output punctuation
    return True


def remove_punctuation_edits(confusion_list):
    """Take confusion_list and remove all edits that convert some character
    so a different output punctuation character."""

    new_confusion_list = []

    for input_str, output_str, weight in confusion_list:

        for i, char in enumerate(input_str):
            if is_punctuation_edit(input_str[i], output_str[i]):
                break
        else:
            new_confusion_list.append((input_str, output_str, weight))
            #print(input_str, output_str, weight)

    return new_confusion_list


def main():
    """Read GT and OCR data, align corresponding text lines, count for
    differing lines ngram pair occurrences for 1grams, 2grams, and 3grams.
    Write confusions frequencies to confusion_<n>.txt and create simple
    error_transducer_<n>.hfst."""

    # read GT data and OCR data (from dta19_reduced)
    path = '../dta19-reduced/traindata/'
    #path = '../dta19-reduced/testdata/'
    gt_dict = helper.create_dict(path, 'gt')

    #frak3_dict = create_dict(path, 'deu-frak3')
    fraktur4_dict = helper.create_dict(path, 'Fraktur4')
    #foo4_dict = create_dict(path, 'foo4')

    # get list of confusion dicts with different context lengths (1 to 3)
    confusion_dicts = get_confusion_dict(gt_dict, fraktur4_dict)

    n = 3

    preserve_punctuation = False

    for n in [1,2,3]: # considered ngrams

        print('n: ', str(n))

        # convert to relative frequencies
        confusion_list = preprocess_confusion_dict(confusion_dicts[n])

        # write confusion to confusion_<n>.txt
        write_frequency_list(confusion_list, 'confusion_' + str(n) + '.txt')
        # reading confusion_<n>.txt is necessary because this changes the
        # data format (convert frequncy from str to float)
        confusion_list = read_frequency_list('confusion_' + str(n) + '.txt')

        print(len(confusion_list))

        if preserve_punctuation:
            confusion_list = remove_punctuation_edits(confusion_list)

        print(len(confusion_list))

        # create (non-complete) error_transducer and optimize it
        error_transducer = transducer_from_list(confusion_list)
        optimize_error_transducer(error_transducer)

        # write transducer to file
        helper.save_transducer('error_transducer_' + str(n) + '.hfst', error_transducer)
        #error_transducer = helper.load_transducer('error_transducer_' + str(n) + '.hfst')

        #print(error_transducer)

if __name__ == '__main__':
    main()
