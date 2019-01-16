from os import listdir

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

def main():
    """
    Read OCR files following the path scheme <directory>/<ID>.<suffix>,
    where each file contains one line of text.
    Correct each line and write output files in same directory with suffix
    specified in output_suffix.
    """

    global lowercase_transducer, lm_transducer, flag_encoder, args, composition
    
    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst batch-processor tool')
    parser.add_argument('directory', metavar='PATH', help='directory for input and output files')
    parser.add_argument('-I', '--input-suffix', metavar='SUF', type=str, default='txt', help='input (OCR) filenames suffix')
    parser.add_argument('-O', '--output-suffix', metavar='SUF', type=str, default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    parser.add_argument('-P', '--punctuation', metavar='MODEL', type=str, choices=['bracket', 'lm', 'preserve'], default='bracket', help='how to model punctuation between words (bracketing rules, inter-word language model, or keep unchanged)')
    parser.add_argument('-W', '--words-per-window', metavar='NUM', type=int, default=3, help='maximum number of words in one window')
    parser.add_argument('-R', '--result-num', metavar='NUM', type=int, default=10, help='result paths per window')
    parser.add_argument('-D', '--composition-depth', metavar='NUM', type=int, default=2, help='number of lexicon words that can be concatenated')
    parser.add_argument('-J', '--rejection-weight', metavar='WEIGHT', type=float, default=1.5, help='transition weight for unchanged input window')
    parser.add_argument('-A', '--apply-lm', action='store_true', default=False, help='also compose with n-gram language model for rescoring')
    parser.add_argument('-Q', '--processes', metavar='NUM', type=int, default=1, help='number of processes to use in parallel')
    parser.add_argument('-L', '--log-level', metavar='LEVEL', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='verbosity of logging output (standard log levels)') # WARN
    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level))
    
    # prepare transducers

    flag_encoder = sw.FlagEncoder()

    #ocr_suffix = 'Fraktur4' # suffix of input files
    #output_suffix = ocr_suffix + '_preserve_2_no_space' # suffix of output files
    #complete_output_suffix = '.' + output_suffix + '.txt'

    # load and construct transducers

    if args.punctuation == 'bracket':
        ## bracketing rules
        error_transducer, lexicon_transducer = sw.load_transducers_bracket(
            'fst/max_error_3_context_23_dta.hfst', # old error model for Fraktur4
            #'fst/max_error_3_context_23_dta19-reduced.' + args.input_suffix[:-3] + 'hfst', # new small error model for chosen input_suffix
            #'fst/max_error_3_context_23_dta19-reduced.Fraktur4.hfst', # new small error model for Fraktur4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.foo4.hfst', # new small error model for foo4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.deu-frak3.hfst', # new small error model for deu-frak3 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.mixed.hfst', # new small mixed error model
            #'fst/max_error_3_context_23_dta19.' + args.input_suffix[:-3] + 'hfst', # new large error model for chosen input_suffix
            #'fst/max_error_3_context_23_dta19.' + args.input_suffix[:-10] + '.hfst', # new large error model for chosen input_suffix (when combining with corRNN)
            #'fst/max_error_3_context_23_dta19.mixed.hfst', # new large mixed error model
            'fst/punctuation_transducer_dta19-reduced.testdata.hfst', # small test corpus punctuation
            #'fst/punctuation_transducer_dta19-reduced.traindata.hfst', # small train corpus punctuation
            #'fst/punctuation_transducer_dta19.testdata.hfst', # large test corpus punctuation
            #'fst/punctuation_transducer_dta19.traindata.hfst', # large train corpus punctuation
            #'fst/lexicon_komplett.again/punctuation_transducer_dta.hfst', # complete corpus punctuation
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst', # small test corpus lexicon
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst', # small train corpus lexicon
            #'fst/lexicon_transducer_dta19.testdata.hfst', # large test corpus lexicon
            #'fst/lexicon_traindata.large/lexicon_transducer_dta.hfst', # large train corpus lexicon
            #'fst/lexicon_komplett.again/lexicon_transducer_dta.hfst', # complete corpus lexicon
            'fst/open_bracket_transducer_dta19-reduced.testdata.hfst', # small test corpus opening brackets
            #'fst/open_bracket_transducer_dta19-reduced.traindata.hfst', # small train corpus opening brackets
            #'fst/open_bracket_transducer_dta19.testdata.hfst', # large test corpus opening brackets
            #'fst/open_bracket_transducer_dta19-traindata.hfst', # large train corpus opening brackets
            #'fst/lexicon_komplett.again/open_bracket_transducer_dta.hfst', # complete corpus opening brackets
            'fst/close_bracket_transducer_dta19-reduced.testdata.hfst', # small test corpus closing brackets
            #'fst/close_bracket_transducer_dta19-reduced.traindata.hfst', # small train corpus closing brackets
            #'fst/close_bracket_transducer_dta19.testdata.hfst', # large test corpus closing brackets
            #'fst/close_bracket_transducer_dta19.traindata.hfst', # large train corpus closing brackets
            #'fst/lexicon_komplett.again/close_bracket_transducer_dta.hfst', # complete corpus closing brackets
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        
    elif args.punctuation == 'lm':
        ## inter-word language model
        error_transducer, lexicon_transducer = sw.load_transducers_inter_word(
            'fst/max_error_3_context_23_dta.hfst', # old error model for Fraktur4
            #'fst/max_error_3_context_23_dta19-reduced.' + args.input_suffix[:-3] + 'hfst', # new small error model for chosen input_suffix
            #'fst/max_error_3_context_23_dta19-reduced.Fraktur4.hfst', # new small error model for Fraktur4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.foo4.hfst', # new small error model for foo4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.deu-frak3.hfst', # new small error model for deu-frak3 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_dta19-reduced.mixed.hfst', # new small mixed error model
            #'fst/max_error_3_context_23_dta19.' + args.input_suffix[:-3] + 'hfst', # new large error model for chosen input_suffix
            #'fst/max_error_3_context_23_dta19.' + args.input_suffix[:-10] + '.hfst', # new large error model for chosen input_suffix (when combining with corRNN)
            #'fst/max_error_3_context_23_dta19.mixed.hfst', # new large mixed error model
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst', # small test corpus lexicon
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst', # small train corpus lexicon
            #'fst/lexicon_transducer_dta19.testdata.hfst', # large test corpus lexicon
            #'fst/lexicon_traindata.large/lexicon_transducer_dta.hfst', # large train corpus lexicon
            #'fst/lexicon_komplett.again/lexicon_transducer_dta.hfst', # complete corpus lexicon
            'fst/left_punctuation.hfst', # old left-of-space/suffix punctuation
            'fst/right_punctuation.hfst', # old right-of-space/prefix punctuation
            flag_encoder,
            words_per_window=args.words_per_window,
            composition_depth=args.composition_depth)
        
    elif args.punctuation == 'preserve':
        ## no punctuation changes
        error_transducer, lexicon_transducer = sw.load_transducers_preserve_punctuation(
            'fst/preserve_punctuation_max_error_3_context_23_dta.hfst', # old error model for Fraktur4
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.' + args.input_suffix[:-3] + 'hfst', # new small error model for chosen input_suffix
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.Fraktur4.hfst', # new small error model for Fraktur4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.foo4.hfst', # new small error model for foo4 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.deu-frak3.hfst', # new small error model for deu-frak3 (when deviating from chosen input_suffix)
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.mixed.hfst', # new small mixed error model
            #'fst/max_error_3_context_23_preserve_punctuation_dta19.' + args.input_suffix[:-3] + 'hfst', # new large error model for chosen input_suffix
            #'fst/max_error_3_context_23_preserve_punctuation_dta19.' + args.input_suffix[:-10] + '.hfst', # new large error model for chosen input_suffix (when combining with corRNN)
            #'fst/max_error_3_context_23_preserve_punctuation_dta19.mixed.hfst', # new large mixed error model
            'fst/any_punctuation_no_space.hfst', # old preserving punctuation
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst', # small test corpus lexicon
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst', # small train corpus lexicon
            #'fst/lexicon_transducer_dta19.testdata.hfst', # large test corpus lexicon
            #'fst/lexicon_traindata.large/lexicon_transducer_dta.hfst', # large train corpus lexicon
            #'fst/lexicon_komplett.again/lexicon_transducer_dta.hfst', # complete corpus lexicon
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        
    
    # prepare Composition Object

    # write lexicon and error transducer files in OpenFST format
    # (cannot use one file for both with OpenFST::Read)
    with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-error') as error_f:
        with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-lexicon') as lexicon_f:
            sw.write_fst(error_f.name, error_transducer)
            sw.write_fst(lexicon_f.name, lexicon_transducer)
            
            composition = pyComposition(error_f.name, lexicon_f.name, args.result_num, args.rejection_weight)
            
            if args.apply_lm:
                lm_file = 'fst/lang_mod_theta_0_000001.mod.modified.hfst'
                lowercase_file = 'fst/lowercase.hfst'
                
                lm_transducer = helper.load_transducer(lm_file)
                lowercase_transducer = helper.load_transducer(lowercase_file)
            
            sw.REJECTION_WEIGHT = args.rejection_weight
    
            # read and process test data
            
            gt_dict = helper.create_dict(args.directory + "/", 'gt.txt')
            ocr_dict = helper.create_dict(args.directory + "/", args.input_suffix)

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
    
    return

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

            with open(args.directory + "/" + basename + "." + suffix, 'w') as f:
                f.write(output_str)
    
    except Exception as e:
        logging.exception('exception for window result of "%s"' % input_str)
        raise e
    
    return basename, input_str, output_str

if __name__ == '__main__':
    main()
