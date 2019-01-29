from os import listdir
import os.path

import logging
import argparse
import tempfile
import multiprocessing as mp
import hfst

from composition import pyComposition
import sliding_window as sw
import helper

# globals (for painless cow-semantic shared memory fork-based multiprocessing)
lowercase_transducer = None
lm_transducer = None
flag_encoder = None
composition = None
args = None


def load_transducers(punctuation, composition_depth, words_per_window):
    global flag_encoder

    if args.punctuation == 'bracket':
        ## bracketing rules
        return sw.load_transducers_bracket(
            'fst/max_error_3_context_23.hfst', # old error model for Fraktur4
            'fst/punctuation_transducer_dta.hfst', # small test corpus punctuation
            'fst/lexicon_transducer_dta.hfst', # small test corpus lexicon
            'fst/open_bracket_transducer_dta.hfst', # small test corpus opening brackets
            'fst/close_bracket_transducer_dta.hfst', # small test corpus closing brackets
            flag_encoder,
            composition_depth=composition_depth,
            words_per_window=words_per_window)
        
    elif args.punctuation == 'lm':
        ## inter-word language model
        return sw.load_transducers_inter_word(
            'fst/max_error_3_context_23.hfst', # old error model for Fraktur4
            'fst/lexicon_transducer_dta.hfst', # small test corpus lexicon
            'fst/left_punctuation.hfst', # old left-of-space/suffix punctuation
            'fst/right_punctuation.hfst', # old right-of-space/prefix punctuation
            flag_encoder,
            composition_depth=composition_depth,
            words_per_window=words_per_window)
        
    elif args.punctuation == 'preserve':
        ## no punctuation changes
        return sw.load_transducers_preserve_punctuation(
            'fst/preserve_punctuation_max_error_3_context_23.hfst', # old error model for Fraktur4
            'fst/any_punctuation_no_space.hfst', # old preserving punctuation
            'fst/lexicon_transducer_dta.hfst', # small test corpus lexicon
            flag_encoder,
            composition_depth=composition_depth,
            words_per_window=words_per_window)

    
def prepare_composition(lexicon_transducer, error_transducer):
    # write lexicon and error transducer files in OpenFST format
    # (cannot use one file for both with OpenFST::Read)
    result = None
    with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-error') as error_f:
        with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-lexicon') as lexicon_f:
            sw.write_fst(error_f.name, error_transducer)
            sw.write_fst(lexicon_f.name, lexicon_transducer)
            
            result = pyComposition(
                error_f.name, lexicon_f.name,
                args.result_num, args.rejection_weight)
    return result


# needs to be global for mp:
def process(basename, input_str):
    global lowercase_transducer, lm_transducer, flag_encoder, args, composition
    
    logging.info('input_str:  %s', input_str)

    try:
        complete_outputs = sw.window_size_1_2(input_str, None, None, flag_encoder, args.result_num, composition)
        
        for i, complete_output in enumerate(complete_outputs):
            if not args.apply_lm:
                complete_output.n_best(1)

            complete_output = sw.remove_flags(complete_output, flag_encoder)

            if args.apply_lm:
                complete_output.output_project()
                # FIXME: should also be composed via OpenFST library (pyComposition)
                complete_output.compose(lowercase_transducer)
                complete_output.compose(lm_transducer)
                complete_output.input_project()

            complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
            output_str = list(complete_paths.items())[0][1][0][0].replace(hfst.EPSILON, '') # really necessary?

            logging.info('output_str: %s', output_str)

            if sw.REJECTION_WEIGHT < 0: # for ROC evaluation: multiple output files
                suffix = args.output_suffix + "." + "rw_" + str(i)
            else:
                suffix = args.output_suffix

            filename = basename + "." + suffix
            with open(os.path.join(args.directory, filename), 'w') as f:
                f.write(output_str)
    
    except Exception as e:
        logging.exception('exception for window result of "%s"' % input_str)
        raise e
    
    return basename, input_str, output_str


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction ocrd-cor-asv-fst batch-processor '
                    'tool')
    parser.add_argument(
        'directory', metavar='PATH',
        help='directory for input and output files')
    parser.add_argument(
        '-I', '--input-suffix', metavar='SUF', type=str, default='txt',
        help='input (OCR) filenames suffix')
    parser.add_argument(
        '-O', '--output-suffix', metavar='SUF', type=str,
        default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    parser.add_argument(
        '-P', '--punctuation', metavar='MODEL', type=str,
        choices=['bracket', 'lm', 'preserve'], default='bracket',
        help='how to model punctuation between words (bracketing rules, '
             'inter-word language model, or keep unchanged)')
    parser.add_argument(
        '-W', '--words-per-window', metavar='NUM', type=int, default=3,
        help='maximum number of words in one window')
    parser.add_argument(
        '-R', '--result-num', metavar='NUM', type=int, default=10,
        help='result paths per window')
    parser.add_argument(
        '-D', '--composition-depth', metavar='NUM', type=int, default=2,
        help='number of lexicon words that can be concatenated')
    parser.add_argument(
        '-J', '--rejection-weight', metavar='WEIGHT', type=float, default=1.5,
        help='transition weight for unchanged input window')
    parser.add_argument(
        '-A', '--apply-lm', action='store_true', default=False,
        help='also compose with n-gram language model for rescoring')
    parser.add_argument(
        '-Q', '--processes', metavar='NUM', type=int, default=1,
        help='number of processes to use in parallel')
    parser.add_argument(
        '-L', '--log-level', metavar='LEVEL', type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='verbosity of logging output (standard log levels)')
    return parser.parse_args()


def main():
    """
    Read OCR files following the path scheme <directory>/<ID>.<suffix>,
    where each file contains one line of text.
    Correct each line and write output files in same directory with suffix
    specified in output_suffix.
    """

    global lowercase_transducer, lm_transducer, flag_encoder, args, composition
    
    args = parse_arguments()

    logging.basicConfig(level=logging.getLevelName(args.log_level))
    
    # prepare transducers
    flag_encoder = sw.FlagEncoder()
    error_transducer, lexicon_transducer = load_transducers(
        args.punctuation, args.composition_depth, args.words_per_window)
    composition = prepare_composition(lexicon_transducer, error_transducer)
            
    if args.apply_lm:
        lm_file = 'fst/lang_mod_theta_0_000001.mod.modified.hfst'
        lowercase_file = 'fst/lowercase.hfst'
        
        lm_transducer = helper.load_transducer(lm_file)
        lowercase_transducer = helper.load_transducer(lowercase_file)
    
    sw.REJECTION_WEIGHT = args.rejection_weight

    # read and process test data
    gt_dict = helper.create_dict(args.directory, 'gt.txt')
    ocr_dict = helper.create_dict(args.directory, args.input_suffix)

    def show_error(exception):
        logging.error(exception)
    
    results = []
    if args.processes > 1:
        with mp.Pool(processes=args.processes) as pool:
            params = list(ocr_dict.items()) #[10:20]
            result = pool.starmap_async(process, params, error_callback=show_error)
            result.wait()
            if result.successful():
                results = result.get()
            else:
                logging.error('error during processing')
                exit(1)
    else:
        results = [process(basename, input_str) \
                   for basename, input_str in ocr_dict.items()]
    
    for i, (basename, input_str, output_str) in enumerate(results):
        print("%03d/%03d: %s" % (i+1, len(ocr_dict), basename))
        print(input_str)
        print(output_str)
        print(gt_dict[basename])
        print()


if __name__ == '__main__':
    main()
