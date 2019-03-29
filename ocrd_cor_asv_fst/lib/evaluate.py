import argparse

# from alignment.sequence import Sequence
# import alignment
# alignment.sequence.GAP_ELEMENT = 0 #"ε"
# from alignment.vocabulary import Vocabulary
# from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner

from difflib import SequenceMatcher # faster (and no memory/stack problems), but no customized distance metrics
matcher = SequenceMatcher(isjunk=None, autojunk=False)
GAP_ELEMENT = 0

import editdistance # fastest (and no memory/stack problems), but no customized distance metrics and no alignment result

from .helper import load_pairs_from_dir, load_pairs_from_file


def print_line(ocr, cor, gt):
    print('OCR:       ', ocr)
    print('Corrected: ', cor)
    print('GT:        ', gt)


def get_best_alignment(l1, l2):
    # scoring = SimpleScoring(2, -1)
    # aligner = StrictGlobalSequenceAligner(scoring, -2)

    # a = Sequence(l1)
    # b = Sequence(l2)

    # # create a vocabulary and encode the sequences
    # vocabulary = Vocabulary()
    # source_seq = vocabulary.encodeSequence(a)
    # target_seq = vocabulary.encodeSequence(b)

    # score = aligner.align(source_seq, target_seq)
    # if score < 5-len(source_seq)/2:
    #     return editdistance.eval(l1, l2), len(l2) # prevent stack/heap overflow with aligner
    
    # _, alignments = aligner.align(source_seq, target_seq, backtrace=True)
    # a = vocabulary.decodeSequenceAlignment(alignments[0]) # best result

    # #print(a)
    global matcher
    matcher.set_seqs(l1, l2)

    alignment1 = []
    for op, l1_begin, l1_end, l2_begin, l2_end in matcher.get_opcodes():
        if op == 'equal':
            alignment1.extend(zip(l1[l1_begin:l1_end],
                                  l2[l2_begin:l2_end]))
        elif op == 'replace': # not really substitution:
            delta = l1_end-l1_begin-l2_end+l2_begin
            #alignment1.extend(zip(l1[l1_begin:l1_end] + [GAP_ELEMENT]*(-delta),
            #                      l2[l2_begin:l2_end] + [GAP_ELEMENT]*(delta)))
            if delta > 0: # replace+delete
                alignment1.extend(zip(l1[l1_begin:l1_end-delta],
                                      l2[l2_begin:l2_end]))
                alignment1.extend(zip(l1[l1_end-delta:l1_end],
                                      [GAP_ELEMENT]*(delta)))
            if delta <= 0: # replace+insert
                alignment1.extend(zip(l1[l1_begin:l1_end],
                                      l2[l2_begin:l2_end+delta]))
                alignment1.extend(zip([GAP_ELEMENT]*(-delta),
                                      l2[l2_end+delta:l2_end]))
        elif op == 'insert':
            alignment1.extend(zip([GAP_ELEMENT]*(l2_end-l2_begin),
                                  l2[l2_begin:l2_end]))
        elif op == 'delete':
            alignment1.extend(zip(l1[l1_begin:l1_end],
                                  [GAP_ELEMENT]*(l1_end-l1_begin)))
        else:
            raise Exception("difflib returned invalid opcode", op, "in", l1, l2)
    return alignment1


def get_adjusted_distance(l1, l2):
    """Calculate distance (as the number of edits) of strings l1 and l2 by aligning them.
    The adjusted length and distance here means that diacritical characters are counted
    as only one character. Thus, for each occurrence of such a character the
    length is reduced by 1."""
    alignment1 = get_best_alignment(l1, l2)
    
    # the following code ensures that diacritical characters are counted as
    # a single character (and not as 2)

    d = 0 # distance

    umlauts = {u"ä": "a", u"ö": "o", u"ü": "u"} # for example
    #umlauts = {}

    source_umlaut = ''
    target_umlaut = ''

    #for source_sym, target_sym in zip(a.first, a.second):
    for source_sym, target_sym in alignment1:

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
                if (source_sym == GAP_ELEMENT and
                    target_sym == u"\u0364"): # diacritical combining e
                    d += 1.0 # umlaut error (umlaut match)
                    #print('source umlaut match', a)
                else:
                    d += 2.0 # two full errors (mismatch)
            elif target_umlaut: # previous target is umlaut non-error
                target_umlaut = False # reset
                if (target_sym == GAP_ELEMENT and
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

    #length_reduction = max(l1.count(u"\u0364"), l2.count(u"\u0364"))
    return d, len(l2) # d, len(a) - length_reduction # distance and adjusted length


def get_precision_recall(ocr, cor, gt):
    """
    Calculate number of true/false positive/negative edits of given OCR vs
    GT and COR vs GT line by aligning them.
    
    Return the true positive, true negative, false positive, false negative
    counts as a tuple.
    """

    def _merge_alignments(al_1, al_2):
        '''
        Merges alignment `al_1` between sequences A, C and `al_2` between
        sequences B, C into a three-way alignment between A, B, C.
        '''
        x1, y1 = next(al_1)
        x2, y2 = next(al_2)
        while True:
            try:
                if y1 == y2 and y1 != GAP_ELEMENT:
                    yield x1, x2, y1
                    x1, y1 = next(al_1)
                    x2, y2 = next(al_2)
                elif y1 == GAP_ELEMENT:
                    yield x1, '', ''
                    x1, y1 = next(al_1)
                elif y2 == GAP_ELEMENT:
                    yield '', x2, ''
                    x2, y2 = next(al_2)
                else:
                    raise RuntimeError(\
                        'Sequence mismatch in three-way alignment.')
            except StopIteration:
                break

    al_ocr = get_best_alignment(ocr, gt)
    al_cor = get_best_alignment(cor, gt)
    
    TP, FP, TN, FN = 0, 0, 0, 0
    for c_ocr, c_cor, c_gt in _merge_alignments(iter(al_ocr), iter(al_cor)):
        is_correct = (c_cor == c_gt)
        is_changed = (c_cor != c_ocr)
        TP += 1 if is_changed and is_correct else 0
        FP += 1 if is_changed and not is_correct else 0
        TN += 1 if not is_changed and is_correct else 0
        FN += 1 if not is_changed and not is_correct else 0

    return (TP, TN, FP, FN)


def compute_total_precision_recall(line_triplets, silent=False):
    TP, TN, FP, FN = 0, 0, 0, 0
    for ocr, cor, gt in line_triplets:
        l_TP, l_TN, l_FP, l_FN = get_precision_recall(ocr, cor, gt)
        TP += l_TP
        TN += l_TN
        FP += l_FP
        FN += l_FN
        if not silent:
            print_line(ocr, cor, gt)
            print("TP: %d / TN: %d / FP: %d / FN: %d" %
                  (l_TP, l_TN, l_FP, l_FN))
            print('precision: %.3f / recall %.3f' %
                  (1 if l_TP+l_FP == 0 else l_TP / (l_TP+l_FP),
                   1 if l_TP+l_FN == 0 else l_TP / (l_TP+l_FN)))
            print()
    return TP, TN, FP, FN


# FIXME: convert to NFC (canonical composition normal form) before
#        perhaps even NFKC (canonical composition compatibility normal form),
#                but GT guidelines require keeping "ſ"
def compute_total_edits_levenshtein(line_triplets, silent=False):
    edits_ocr, len_ocr, edits_cor, len_cor = 0, 0, 0, 0
    for ocr, cor, gt in line_triplets:
        edits_ocr_line, len_ocr_line = editdistance.eval(ocr, gt), len(gt)
        edits_cor_line, len_cor_line = editdistance.eval(cor, gt), len(gt)
        edits_ocr += edits_ocr_line
        len_ocr   += len_ocr_line
        edits_cor += edits_cor_line
        len_cor   += len_cor_line
        if not silent:
            print_line(ocr, cor, gt)
            print('CER OCR:       ', edits_ocr_line / len_ocr_line)
            print('CER Corrected: ', edits_cor_line / len_cor_line)
    return edits_ocr, len_ocr, edits_cor, len_cor


def compute_total_edits_combining_e_umlauts(line_triplets, silent=False):
    edits_ocr, len_ocr, edits_cor, len_cor = 0, 0, 0, 0
    for ocr, cor, gt in line_triplets:
        edits_ocr_line, len_ocr_line = get_adjusted_distance(ocr, gt)
        edits_cor_line, len_cor_line = get_adjusted_distance(cor, gt)
        edits_ocr += edits_ocr_line
        len_ocr   += len_ocr_line
        edits_cor += edits_cor_line
        len_cor   += len_cor_line
        if not silent:
            print_line(ocr, cor, gt)
            print('CER OCR:       ', edits_ocr_line / len_ocr_line)
            print('CER Corrected: ', edits_cor_line / len_cor_line)
    return edits_ocr, len_ocr, edits_cor, len_cor


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction batch evaluation ocrd-cor-asv-fst')
    parser.add_argument(
        '-d', '--directory', metavar='PATH',
        help='directory for GT, input, and output files')
    parser.add_argument(
        '-I', '--input-suffix', metavar='SUF', type=str, default=None,
        help='input (OCR) filenames suffix')
    parser.add_argument(
        '-i', '--input-file', metavar='FILE', type=str, default=None,
        help='file with input (OCR) data')
    parser.add_argument(
        '-O', '--output-suffix', metavar='SUF', type=str,
        default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    parser.add_argument(
        '-o', '--output-file', metavar='FILE', type=str, default=None,
        help='file with output (corrected) data')
    parser.add_argument(
        '-G', '--gt-suffix', metavar='SUF', type=str,
        default='gt.txt', help='ground truth filenames suffix')
    parser.add_argument(
        '-g', '--gt-file', metavar='FILE', type=str, default=None,
        help='file with ground truth data')
    parser.add_argument(
        '-M', '--metric', metavar='TYPE', type=str,
        choices=['Levenshtein', 'combining-e-umlauts', 'precision-recall'],
        default='combining-e-umlauts', help='distance metric to apply')
    parser.add_argument(
        '-S', '--silent', action='store_true', default=False,
        help='do not show data, only aggregate')
    return parser.parse_args()


def main():
    """
    Read GT files, OCR files, and corrected files 
    following the path scheme <directory>/<ID>.<suffix>,
    where each file contains one line of text, 
    for measuring and comparing the character error rate (CER).
    
    For GT files, the suffix is given in <gt_suffix>.
    For OCR files, the suffix is given in <input_suffix>.
    For corrected files, the suffix is given in <output_suffix>.
    
    Align corresponding lines (with same ID) from GT, OCR, and correction,
    and measure their edit distance and CER.
    """

    args = parse_arguments()

    # check the validity of parameters specifying input/output
    if args.input_file is None and \
            (args.input_suffix is None or args.directory is None):
        raise RuntimeError('No input data supplied! You have to specify either'
                           ' -i or -I and the data directory.')
    if args.output_file is None and \
            (args.output_suffix is None or args.directory is None):
        raise RuntimeError('No output file speficied! You have to specify '
                           'either -o or -O and the data directory.')
    if args.gt_file is None and \
            (args.gt_suffix is None or args.directory is None):
        raise RuntimeError('No ground truth file speficied! You have to '
                           'specify either -g or -G and the data directory.')
    
    # read the test data
    ocr_dict = dict(load_pairs_from_file(args.input_file)) \
               if args.input_file is not None \
               else dict(load_pairs_from_dir(args.directory, args.input_suffix))
    cor_dict = dict(load_pairs_from_file(args.output_file)) \
               if args.output_file is not None \
               else dict(load_pairs_from_dir(args.directory, args.output_suffix))
    gt_dict = dict(load_pairs_from_file(args.gt_file)) \
              if args.gt_file is not None \
              else dict(load_pairs_from_dir(args.directory, args.gt_suffix))
    line_triplets = \
        ((ocr_dict[key].strip(), cor_dict[key].strip(), gt_dict[key].strip()) \
         for key in gt_dict)

    if args.metric == 'precision-recall':
        TP, TN, FP, FN = compute_total_precision_recall(
            line_triplets, silent=args.silent)
        precision = 1 if TP+FP==0 else TP/(TP+FP)
        recall = 1 if TP+FN==0 else TP/(TP+FN)
        f1 = 2*TP/(2*TP+FP+FN)
        tpr = recall                                # "sensitivity"
        fpr = 0 if FP+TN==0 else FP/(FP+TN)         # "overcorrection rate"
        auc = 0.5*tpr*fpr+tpr*(1-fpr)+0.5*(1-tpr)*(1-fpr)
        print('Aggregate precision: %.3f / recall: %.3f / F1: %.3f' %
              (precision, recall, f1))
        print('Aggregate true-positive-rate: %.3f '
              '/ false-positive-rate: %.3f / AUC: %.3f' %
              (tpr, fpr, auc))

    elif args.metric == 'Levenshtein':
        edits_ocr, len_ocr, edits_cor, len_cor = \
            compute_total_edits_levenshtein(line_triplets, silent=args.silent)
        print('Aggregate CER OCR:       ', edits_ocr / len_ocr)
        print('Aggregate CER Corrected: ', edits_cor / len_cor)

    elif args.metric == 'combining-e-umlauts':
        edits_ocr, len_ocr, edits_cor, len_cor = \
            compute_total_edits_combining_e_umlauts(
                line_triplets, silent=args.silent)
        print('Aggregate CER OCR:       ', edits_ocr / len_ocr)
        print('Aggregate CER Corrected: ', edits_cor / len_cor)


if __name__ == '__main__':
    main()

