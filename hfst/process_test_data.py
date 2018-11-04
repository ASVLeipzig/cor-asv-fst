from os import listdir

import argparse
import multiprocessing as mp
import hfst

import sliding_window as sw
from composition import pyComposition
import helper


def main():
    """
    Read OCR files following the path scheme <directory>/<ID>.<suffix>,
    where each file contains one line of text.
    Correct each line and write output files in same directory with suffix
    specified in output_suffix.
    """

    parser = argparse.ArgumentParser(description='OCR post-correction batch tool ocrd-cor-asv-fst')
    parser.add_argument('directory', metavar='PATH', help='directory for input and output files')
    parser.add_argument('-I', '--input-suffix', metavar='ISUF', type=str, default='txt', help='input (OCR) filenames suffix')
    parser.add_argument('-O', '--output-suffix', metavar='OSUF', type=str, default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    parser.add_argument('-W', '--words-per-window', metavar='WORDS', type=int, default=3, help='maximum number of words in one window')
    parser.add_argument('-R', '--result-num', metavar='RESULTS', type=int, default=10, help='result paths per window')
    parser.add_argument('-D', '--composition-depth', metavar='DEPTH', type=int, default=2, help='number of lexicon words that can be concatenated')
    args = parser.parse_args()
    
    # prepare transducers

    flag_encoder = sw.FlagEncoder()

    #ocr_suffix = 'Fraktur4' # suffix of input files
    #output_suffix = ocr_suffix + '_preserve_2_no_space' # suffix of output files
    #complete_output_suffix = '.' + output_suffix + '.txt'

    # load and construct transducers

    # bracket model

    #error_transducer, lexicon_transducer =\
    #    load_transducers_bracket(\
    #    'transducers/max_error_3_context_23_dta.hfst',\
    #    'transducers/punctuation_transducer_dta.hfst',\
    #    'transducers/lexicon_transducer_dta.hfst',\
    #    'transducers/open_bracket_transducer_dta.hfst',\
    #    'transducers/close_bracket_transducer_dta.hfst',\
    #    flag_encoder,\
    #    composition_depth = composition_depth,\
    #    words_per_window = words_per_window)

    # no punctuation changes

    error_transducer, lexicon_transducer =\
        sw.load_transducers_preserve_punctuation(\
        'fst/preserve_punctuation_max_error_3_context_23.hfst',\
        'fst/any_punctuation_no_space.hfst',\
        'fst/lexicon_transducer_dta.hfst',\
        flag_encoder,\
        composition_depth = args.composition_depth,\
        words_per_window = args.words_per_window)

    # prepare Composition Object

    error_filename = 'error.ofst'
    lexicon_filename = 'lexicon.ofst'
    error_filename_b = b'error.ofst'
    lexicon_filename_b = b'lexicon.ofst'

    for filename, fst in [(error_filename, error_transducer), (lexicon_filename, lexicon_transducer)]:
        out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        out.write(fst)
        out.flush()
        out.close()

    composition = pyComposition(error_filename_b, lexicon_filename_b, args.result_num)

    # read and process test data

    #path = '../../../daten/dta19-reduced/testdata/'

    gt_dict = helper.create_dict(args.directory + "/", 'gt.txt')
    ocr_dict = helper.create_dict(args.directory + "/", args.input_suffix)

    
    for key, value in list(ocr_dict.items()):#[10:20]:

        input_str = value

        print(key)
        print(value)
        print(gt_dict[key])

        complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, args.result_num, composition)
        complete_output.n_best(1)
        complete_output = sw.remove_flags(hfst.HfstBasicTransducer(complete_output), flag_encoder)
        complete_output = hfst.HfstTransducer(complete_output)
        complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
        output_str = list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', '')

        print(output_str)
        print()

        with open(args.directory + "/" + key + "." + args.output_suffix, 'w') as f:
            f.write(output_str)

    return


if __name__ == '__main__':
    main()
