from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = "ε"
# TODO: GAP_ELEMENT ε can cause problems when handling greek text.
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

from functools import reduce

import helper


def get_adjusted_distance(l1, l2):
    """Calculate distance (as the number of edits) of strings l1 and l2 by aligning them.
    The adjusted length and distance here means that diacritical characters are counted
    as only one character. Thus, for each occurrence of such a character the
    length is reduced by 1."""

    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)

    a = Sequence(l1)
    b = Sequence(l2)

    # create a vocabulary and encode the sequences
    vocabulary = Vocabulary()
    source_seq = vocabulary.encodeSequence(a)
    target_seq = vocabulary.encodeSequence(b)

    _, alignments = aligner.align(source_seq, target_seq, backtrace=True)
    a = vocabulary.decodeSequenceAlignment(alignments[0]) # best result

    #print(a)


    # the following code ensures that diacritical characters are counted as
    # a single character (and not as 2)

    d = 0 # distance
    length_reduction = max(l1.count(u"\u0364"), l2.count(u"\u0364"))

    #umlauts = {u"ä": "a", u"ö": "o", u"ü": "u"} # for example
    umlauts = {"a": u"ä", "o": u"ö", "u": u"ü"} # for example

    source_umlaut = ''
    target_umlaut = ''

    for source_sym, target_sym in zip(a.first, a.second):

        #print(source_sym, target_sym)

        if source_sym == target_sym:
            if source_umlaut: # previous source is umlaut non-error
                source_umlaut = False # reset
                d += 1 # one full error (mismatch)
            elif target_umlaut: # previous target is umlaut non-error
                target_umlaut = False # reset
                d += 1 # one full error (mismatch)

        else:
            if source_umlaut: # previous source is umlaut non-error
                if source_sym == u"\u0364" and\
                   target_sym == umlauts.get(source_umlaut): # diacritical combining e
                    d += 1.0 # umlaut error (match)
                elif source_sym == u"\u0364":
                    d += 1.0 # one error, because diacritical and other character (mismatch)
                else:
                    d += 2.0 # two full errors (mismatch)
                source_umlaut = '' # reset

            elif target_umlaut: # previous target is umlaut non-error
                if target_sym == u"\u0364" and\
                   source_sym == umlauts.get(target_umlaut): # diacritical combining e
                    d += 1.0 # umlaut error (match)
                elif target_sym == u"\u0364":
                    d += 1.0 # one error, because diacritical and other character (mismatch)
                else:
                    d += 2.0 # two full errors (mismatch)
                target_umlaut = '' # reset

            elif source_sym == alignment.sequence.GAP_ELEMENT and\
                target_sym in list(umlauts.keys()):
                target_umlaut = target_sym # umlaut non-error
                #print('set target_umlaut')

            elif target_sym == alignment.sequence.GAP_ELEMENT and\
                source_sym in list(umlauts.keys()):
                source_umlaut = source_sym # umlaut non-error
                #print('set source_umlaut')

            else:
                d += 1 # one full error

    if source_umlaut or target_umlaut: # previous umlaut error
        d += 1 # one full error

    return d, len(a) - length_reduction # distance and adjusted length


def get_adjusted_cer(l1, l2):
    """Calculate the character error rate of l1 and l2."""

    distance, length = get_adjusted_distance(l1, l2)

    # each string has 8 filling characters on each side to ensure alignment
    length = length-16
    #print(length)

    return distance / length, length


def main():
    """Read GT files, OCR files, and corrected files for measuring
    and comparing the character error rate (CER).
    They are of the form path/<ID>.<suffix>.txt.
    For GT files, the suffix is gt, OCR suffic is stored in ocr_suffix and
    the suffix for the files that need to be compared is corrected_suffix.
    Corresponding (same ID) OCR and corrected lines are aligned to the GT
    lines and distance and CER are measured."""

    #l1 = '########Mit unendlich ſuͤßem Sehnen########'
    #l2 = '########Mit unendlich ſüßem Sehnen########'
    #print(align_lines(l1, l2))
    #print(get_adjusted_distance(l1, l2))
    #print(get_adjusted_percent_identity(l1, l2))

    # read testdata
    path = '../../dta19-reduced/testdata/'

    ocr_suffix = 'Fraktur4'
    corrected_suffix = 'Fraktur4_preserve_2_no_space'

    ocr_dict = helper.create_dict(path, ocr_suffix)
    gt_dict = helper.create_dict(path, 'gt')
    corrected_dict = helper.create_dict(path, corrected_suffix)

    cer_list_ocr = []
    cer_list_corrected = []

    for key in corrected_dict.keys():

        # padding characters at each side to ensure alignment
        ocr_line = '########' + ocr_dict[key].strip() + '########'
        gt_line = '########' + gt_dict[key].strip() + '########'
        corrected_line = '########' +  corrected_dict[key].strip() + '########'

        print('OCR:  ', ocr_line)
        print('GT:        ', gt_line)
        print('Corrected: ', corrected_line)

        # get character error rate of OCR and corrected text
        cer_ocr, ocr_len = get_adjusted_cer(ocr_line, gt_line)
        cer_corrected, corrected_len = get_adjusted_cer(corrected_line, gt_line)

        print('CER OCR:  ', cer_ocr)
        print('CER Corrected: ', cer_corrected)

        cer_list_ocr.append((cer_ocr, ocr_len))
        cer_list_corrected.append((cer_corrected, corrected_len))

    summed_chars_ocr = reduce(lambda x,y: x + y[1], cer_list_ocr, 0)
    summed_weighted_cer_ocr = reduce(lambda x,y: x + (y[0] * (y[1] / summed_chars_ocr)), cer_list_ocr, 0)

    print('Summed CER OCR:  ', summed_weighted_cer_ocr)

    summed_chars_corrected = reduce(lambda x,y: x + y[1], cer_list_corrected, 0)
    summed_weighted_cer_corrected = reduce(lambda x,y: x + (y[0] * (y[1] / summed_chars_corrected)), cer_list_corrected, 0)

    print('Summed CER Corrected: ', summed_weighted_cer_corrected)


if __name__ == '__main__':
    main()
