from os import listdir

import hfst

import sliding_window as sw
from composition import pyComposition
import helper


def main():
    """Read OCR files of the form path/<ID>.<suffix>.txt.
    Each file contains one line of text.
    Correct each line and write output files in same directory with suffix
    specified in output_suffix."""

    # prepare transducers

    words_per_window = 3 # maximum number of words in one window
    result_num = 10 # result paths per window
    composition_depth = 2 # number of lexicon words that can be concatenated

    flag_encoder = sw.FlagEncoder()

    ocr_suffix = 'Fraktur4' # suffix of input files
    output_suffix = 'Fraktur4_preserve_2_no_space' # suffix of output files

    complete_output_suffix = '.' + output_suffix + '.txt'

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
        'transducers/preserve_punctuation/max_error_3_context_23.hfst',\
        'transducers/any_punctuation_no_space.hfst',\
        'transducers/lexicon_transducer_dta.hfst',\
        flag_encoder,\
        composition_depth = composition_depth,\
        words_per_window = words_per_window)

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

    composition = pyComposition(error_filename_b, lexicon_filename_b, result_num)

    # read and process test data

    path = '../../dta19-reduced/testdata/'

    gt_dict = helper.create_dict(path, 'gt')
    ocr_dict = helper.create_dict(path, 'Fraktur4')

    for key, value in list(ocr_dict.items()):#[10:20]:

        input_str = value

        print(key)
        print(value)
        print(gt_dict[key])

        complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num, composition)
        complete_output.n_best(1)
        complete_output = sw.remove_flags(hfst.HfstBasicTransducer(complete_output), flag_encoder)
        complete_output = hfst.HfstTransducer(complete_output)
        complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
        output_str = list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', '')

        print(output_str)
        print()

        with open(path + key + complete_output_suffix, 'w') as f:
            f.write(output_str)

    return


if __name__ == '__main__':
    main()
