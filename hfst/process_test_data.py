from os import listdir

import argparse
import multiprocessing as mp
import hfst

from composition import pyComposition
import sliding_window as sw
import helper

# globals (for painless cow-semantic shared memory fork-based multiprocessing)
error_transducer = None
lexicon_transducer = None
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

    global error_transducer, lexicon_transducer, lowercase_transducer, lm_transducer, flag_encoder, args, composition
    
    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst batch-processor tool')
    parser.add_argument('directory', metavar='PATH', help='directory for input and output files')
    parser.add_argument('-I', '--input-suffix', metavar='SUF', type=str, default='txt', help='input (OCR) filenames suffix')
    parser.add_argument('-O', '--output-suffix', metavar='SUF', type=str, default='cor-asv-fst.txt', help='output (corrected) filenames suffix')
    parser.add_argument('-P', '--punctuation', metavar='MODEL', type=str, choices=['bracket', 'lm', 'preserve'], default='bracket', help='how to model punctuation between words (bracketing rules, inter-word language model, or keep unchanged)')
    parser.add_argument('-W', '--words-per-window', metavar='NUM', type=int, default=3, help='maximum number of words in one window')
    parser.add_argument('-R', '--result-num', metavar='NUM', type=int, default=10, help='result paths per window')
    parser.add_argument('-D', '--composition-depth', metavar='NUM', type=int, default=2, help='number of lexicon words that can be concatenated')
    parser.add_argument('-J', '--rejection-weight', metavar='WEIGHT', type=float, default=1.5, help='transition weight for unchanged input window')
    parser.add_argument('-L', '--apply-lm', action='store_true', default=False, help='also compose with n-gram language model for rescoring')
    parser.add_argument('-Q', '--processes', metavar='NUM', type=int, default=1, help='number of processes to use in parallel')
    args = parser.parse_args()
    
    # prepare transducers

    flag_encoder = sw.FlagEncoder()

    #ocr_suffix = 'Fraktur4' # suffix of input files
    #output_suffix = ocr_suffix + '_preserve_2_no_space' # suffix of output files
    #complete_output_suffix = '.' + output_suffix + '.txt'

    # load and construct transducers

    if args.punctuation == 'bracket':
        ## bracketing rules
        error_transducer, lexicon_transducer = sw.load_transducers_bracket(
            'fst/max_error_3_context_23_dta.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.' + args.input_suffix[:-3] + 'hfst',
            'fst/punctuation_transducer_dta.hfst',
            'fst/lexicon_transducer_dta.hfst',
            'fst/open_bracket_transducer_dta.hfst',
            'fst/close_bracket_transducer_dta.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        
    elif args.punctuation == 'lm':
        ## inter-word language model
        error_transducer, lexicon_transducer = sw.load_transducers_inter_word(
            'fst/max_error_3_context_23_dta.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.' + args.input_suffix[:-3] + 'hfst',
            'fst/lexicon_transducer_dta.hfst',
            'fst/left_punctuation.hfst',
            'fst/right_punctuation.hfst',
            flag_encoder,
            words_per_window=args.words_per_window,
            composition_depth=args.composition_depth)
        
    elif args.punctuation == 'preserve':
        ## no punctuation changes
        error_transducer, lexicon_transducer = sw.load_transducers_preserve_punctuation(
            'fst/preserve_punctuation_max_error_3_context_23.hfst',
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.' + args.input_suffix[:-3] + 'hfst',
            'fst/any_punctuation_no_space.hfst',
            'fst/lexicon_transducer_dta.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        
    
    # prepare Composition Object

    error_filename = u'error.ofst'
    lexicon_filename = u'lexicon.ofst'

    for filename, fst in [(error_filename, error_transducer), (lexicon_filename, lexicon_transducer)]:
        out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        out.write(fst)
        out.flush()
        out.close()

    composition = pyComposition(error_filename.encode('utf-8'), lexicon_filename.encode('utf-8'), args.result_num)

    if args.apply_lm:
        lm_file = 'fst/lang_mod_theta_0_000001.mod.modified.hfst'
        lowercase_file = 'fst/lowercase.hfst'
        
        lm_transducer = helper.load_transducer(lm_file)
        lowercase_transducer = helper.load_transducer(lowercase_file)

    sw.REJECTION_WEIGHT = args.rejection_weight
    
    # read and process test data

    #path = '../../../daten/dta19-reduced/testdata/'

    gt_dict = helper.create_dict(args.directory + "/", 'gt.txt')
    ocr_dict = helper.create_dict(args.directory + "/", args.input_suffix)
    
    results = []
    with mp.Pool(processes=args.processes) as pool:
        params = list(ocr_dict.items()) #[10:20]
        results = pool.starmap(process, params)
        
    for basename, input_str, output_str in results:
        print(basename)
        print(input_str)
        print(output_str)
        print(gt_dict[basename])
        print()
    
    return

# needs to be global for mp:
def process(basename, input_str):
    global error_transducer, lexicon_transducer, lowercase_transducer, lm_transducer, flag_encoder, args, composition
    
    complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, args.result_num, composition)
    if not args.apply_lm:
        complete_output.n_best(1)
        
    complete_output = sw.remove_flags(hfst.HfstBasicTransducer(complete_output), flag_encoder)
    complete_output = hfst.HfstTransducer(complete_output)
    
    if args.apply_lm:
        complete_output.output_project()
        # FIXME: should also be composed via OpenFST library (pyComposition)
        complete_output.compose(lowercase_transducer)
        complete_output.compose(lm_transducer)
    
    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
    output_str = list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', '')
    
    with open(args.directory + "/" + basename + "." + args.output_suffix, 'w') as f:
        f.write(output_str)
    
    return basename, input_str, output_str

if __name__ == '__main__':
    main()
