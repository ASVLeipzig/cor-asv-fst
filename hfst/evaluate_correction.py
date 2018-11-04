import argparse

from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = 0 #"ε"
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner

import editdistance # faster (and no memory/stack problems), but no customized distance metrics

import helper


def get_adjusted_distance(l1, l2):
    """Calculate distance (as the number of edits) of strings l1 and l2 by aligning them.
    The adjusted length and distance here means that diacritical characters are counted
    as only one character. Thus, for each occurrence of such a character the
    length is reduced by 1."""

    scoring = SimpleScoring(2, -1)
    aligner = StrictGlobalSequenceAligner(scoring, -2)

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

    umlauts = {u"ä": "a", u"ö": "o", u"ü": "u"} # for example
    #umlauts = {}

    source_umlaut = ''
    target_umlaut = ''

    for source_sym, target_sym in zip(a.first, a.second):

        #print(source_sym, target_sym)

        if source_sym == target_sym:
            if source_umlaut: # previous source is umlaut non-error
                source_umlaut = False # reset
                d += 1.0 # one full error (mismatch)
            elif target_umlaut: # previous target is umlaut non-error
                target_umlaut = False # reset
                d += 1.0 # one full error (mismatch)
        else:
            if source_umlaut: # previous source is umlaut non-error
                source_umlaut = False # reset
                if (source_sym == alignment.sequence.GAP_ELEMENT and
                    target_sym == u"\u0364"): # diacritical combining e
                    d += 1.0 # umlaut error (umlaut match)
                    #print('source umlaut match', a)
                else:
                    d += 2.0 # two full errors (mismatch)
            elif target_umlaut: # previous target is umlaut non-error
                target_umlaut = False # reset
                if (target_sym == alignment.sequence.GAP_ELEMENT and
                    source_sym == u"\u0364"): # diacritical combining e
                    d += 1.0 # umlaut error (umlaut match)
                    #print('target umlaut match', a)
                else:
                    d += 2.0 # two full errors (mismatch)
            elif source_sym in umlauts and umlauts[source_sym] == target_sym:
                source_umlaut = True # umlaut non-error
            elif target_sym in umlauts and umlauts[target_sym] == source_sym:
                target_umlaut = True # umlaut non-error
            else:
                d += 1.0 # one full error (non-umlaut mismatch)
    if source_umlaut or target_umlaut: # previous umlaut error
        d += 1.0 # one full error

    return d, len(l2) # d, len(a) - length_reduction # distance and adjusted length


def main():
    """
    Read GT files, OCR files, and corrected files 
    following the path scheme <directory>/<ID>.<suffix>,
    where each file contains one line of text, 
    for measuring and comparing the character error rate (CER).
    
    For GT files, the suffix is fixed as 'gt.txt'.
    For OCR files, the suffix is given in <input_suffix>.
    For corrected files, the suffix is given in <output_suffix>.
    
    Align corresponding lines (with same ID) from GT, OCR, and correction,
    and measure their edit distance and CER.
    """
    
    parser = argparse.ArgumentParser(description='OCR post-correction batch evaluation ocrd-cor-asv-fst')
    parser.add_argument('directory', metavar='PATH', help='directory for GT, input, and output files')
    parser.add_argument('-I', '--input-suffix', metavar='ISUF', type=str, default='txt', help='input (OCR) filenames suffix')
    parser.add_argument('-O', '--output-suffix', metavar='OSUF', type=str, default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    args = parser.parse_args()
    
    #l1 = '########Mit unendlich ſuͤßem Sehnen########'
    #l2 = '########Mit unendlich ſüßem Sehnen########'
    #print(align_lines(l1, l2))
    #print(get_adjusted_distance(l1, l2))
    #print(get_adjusted_percent_identity(l1, l2))
    
    # read testdata
    #path = '../../../daten/dta19-reduced/testdata/'
    #ocr_suffix = 'Fraktur4'
    #corrected_suffix = ocr_suffix + '_preserve_2_no_space'
    
    ocr_dict = helper.create_dict(args.directory + "/", args.input_suffix)
    gt_dict = helper.create_dict(args.directory + "/", 'gt.txt')
    cor_dict = helper.create_dict(args.directory + "/", args.output_suffix)
    
    edits_ocr, edits_cor = 0, 0
    len_ocr, len_cor = 0,0
    
    for key in cor_dict.keys():
        
        # padding characters at each side to ensure alignment
        #gt_line = '########' + gt_dict[key].strip() + '########'
        #ocr_line = '########' + ocr_dict[key].strip() + '########'
        #cor_line = '########' +  cor_dict[key].strip() + '########'
        gt_line = gt_dict[key].strip()
        ocr_line = ocr_dict[key].strip()
        cor_line = cor_dict[key].strip()
        
        print('OCR:       ', ocr_line)
        print('Corrected: ', cor_line)
        print('GT:        ', gt_line)
        
        # get character error rate of OCR and corrected text
        edits_ocr_line, len_ocr_line = get_adjusted_distance(ocr_line, gt_line) #editdistance.eval(ocr_line, gt_line), len(gt_line)
        edits_cor_line, len_cor_line = get_adjusted_distance(cor_line, gt_line) #editdistance.eval(cor_line, gt_line), len(gt_line)
        print('CER OCR:       ', edits_ocr_line / len_ocr_line)
        print('CER Corrected: ', edits_cor_line / len_cor_line)
        
        edits_ocr += edits_ocr_line
        edits_cor += edits_cor_line
        len_ocr += len_ocr_line
        len_cor += len_cor_line
        
    
    print('Aggregate CER OCR:       ', edits_ocr / len_ocr)
    print('Aggregate CER Corrected: ', edits_cor / len_cor)


if __name__ == '__main__':
    main()
