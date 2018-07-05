from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = "ε"
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

from functools import reduce

import process_test_data as ptd


def get_percent_identity(alignment):

    a1 = alignment[0]
    a2 = alignment[1]

    identity_count = 0;
    char_count = len(a1)

    for i, char in enumerate(a1):
        if a1[i] == a2[i]:
            identity_count += 1

    return identity_count / char_count



def align_lines(l1, l2):

    a = Sequence(l1)
    b = Sequence(l2)

    # Create a vocabulary and encode the sequences.
    v = Vocabulary()
    aEncoded = v.encodeSequence(a)
    bEncoded = v.encodeSequence(b)

    # Create a scoring and align the sequences using global aligner.
    scoring = SimpleScoring(2, -1)
    aligner = GlobalSequenceAligner(scoring, -2)
    score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

    for encoded in encodeds:
        alignment = v.decodeSequenceAlignment(encoded)
        alignment = list(alignment[8:-8])
        cer = 1 - get_percent_identity(alignment)

    return cer, len(alignment[0])


def main():

    path = '../dta19-reduced/testdata/'

    fraktur4_dict = ptd.create_dict(path, 'Fraktur4')
    gt_dict = ptd.create_dict(path, 'gt')
    corrected_dict = ptd.create_dict(path, 'Fraktur4_corrected')

    cer_list_fraktur4 = []
    cer_list_corrected = []

    for key in corrected_dict.keys():

        fraktur4_line = '########' + fraktur4_dict[key].strip() + '########'
        gt_line = '########' + gt_dict[key].strip() + '########'
        corrected_line = '########' +  corrected_dict[key].strip() + '########'

        print('Fraktur4:  ', fraktur4_line)
        print('GT:        ', gt_line)
        print('Corrected: ', corrected_line)

        cer_fraktur4, fraktur4_len = align_lines(fraktur4_line, gt_line)
        cer_corrected, corrected_len = align_lines(corrected_line, gt_line)

        print('CER Fraktur4:  ', cer_fraktur4)
        print('CER Corrected: ', cer_corrected)

        cer_list_fraktur4.append((cer_fraktur4, fraktur4_len))
        cer_list_corrected.append((cer_corrected, corrected_len))

    summed_chars_fraktur4 = reduce(lambda x,y: x + y[1], cer_list_fraktur4, 0)
    summed_weighted_cer_fraktur4 = reduce(lambda x,y: x + (y[0] * (y[1] / summed_chars_fraktur4)), cer_list_fraktur4, 0)

    print('Summed CER Fraktur4:  ', summed_weighted_cer_fraktur4)

    summed_chars_corrected = reduce(lambda x,y: x + y[1], cer_list_corrected, 0)
    summed_weighted_cer_corrected = reduce(lambda x,y: x + (y[0] * (y[1] / summed_chars_corrected)), cer_list_corrected, 0)

    print('Summed CER Corrected: ', summed_weighted_cer_corrected)




if __name__ == '__main__':
    main()
