from os import listdir
import os.path

import logging
import argparse
import tempfile
import multiprocessing as mp
import hfst

from extensions.composition import pyComposition
import sliding_window as sw
import helper

# globals (for painless cow-semantic shared memory fork-based multiprocessing)
model = {}
gl_config = {}
    

def prepare_composition(lexicon_transducer, error_transducer, result_num, rejection_weight):
    # write lexicon and error transducer files in OpenFST format
    # (cannot use one file for both with OpenFST::Read)
    result = None
    with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-error') as error_f:
        with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-lexicon') as lexicon_f:
            helper.save_transducer(
                error_f.name, error_transducer, hfst_format=False,
                type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            helper.save_transducer(
                lexicon_f.name, lexicon_transducer, hfst_format=False,
                type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            
            result = pyComposition(
                error_f.name, lexicon_f.name,
                result_num, rejection_weight)
    return result


def prepare_model(punctuation_method, **kwargs):
    result = { 'flag_encoder' : sw.FlagEncoder() }

    transducers = {
        'flag_encoder' : result['flag_encoder'],
        'lexicon' : helper.load_transducer('fst/lexicon_transducer_dta.hfst')
    }
    if punctuation_method == 'bracket':
        transducers['error'] = helper.load_transducer(
            'fst/error.hfst')
        transducers['punctuation'] = helper.load_transducer(
            'fst/punctuation_transducer_dta.hfst')
        transducers['open_bracket'] = helper.load_transducer(
            'fst/open_bracket_transducer_dta.hfst')
        transducers['close_bracket'] = helper.load_transducer(
            'fst/close_bracket_transducer_dta.hfst')
    elif punctuation_method == 'lm':
        transducers['error'] = helper.load_transducer(
            'fst/max_error_3_context_23.hfst')
        transducers['punctuation_left'] = helper.load_transducer(
            'fst/left_punctuation.hfst')
        transducers['punctuation_right'] = helper.load_transducer(
            'fst/right_punctuation.hfst')
    elif punctuation_method == 'preserve':
        transducers['error'] = helper.load_transducer(
            'fst/preserve_punctuation_max_error_3_context_23.hfst')
        transducers['punctuation'] = helper.load_transducer(
            'fst/any_punctuation_no_space.hfst')

    error_tr, lexicon_tr = sw.build_model(transducers,
        punctuation_method=punctuation_method,
        composition_depth=kwargs['composition_depth'],
        words_per_window=kwargs['words_per_window'])
    result['composition'] = prepare_composition(
        lexicon_tr, error_tr, kwargs['result_num'], kwargs['rejection_weight'])

    if kwargs['apply_lm']:
        result['lm_transducer'] = helper.load_transducer(
            'fst/lang_mod_theta_0_000001.mod.modified.hfst')
        result['lowercase_transducer'] = helper.load_transducer(
            'fst/lowercase.hfst')

    return result


# needs to be global for multiprocessing
def correct_string(basename, input_str):
    global model, gl_config

    def _apply_lm(output_tr):
        output_tr.output_project()
        # FIXME: should also be composed via OpenFST library (pyComposition)
        output_tr.compose(model['lowercase_transducer'])
        output_tr.compose(model['lm_transducer'])
        output_tr.input_project()

    def _save_output_str(output_str, i):
        if sw.REJECTION_WEIGHT < 0: # for ROC evaluation: multiple output files
            suffix = gl_config['output_suffix'] + "." + "rw_" + str(i)
        else:
            suffix = gl_config['output_suffix']

        filename = basename + "." + suffix
        with open(os.path.join(gl_config['directory'], filename), 'w') as f:
            f.write(output_str)

    def _output_tr_to_string(tr):
        paths = hfst.HfstTransducer(tr).extract_paths(
            max_number=1, max_cycles=0)
        return list(paths.items())[0][1][0][0]\
                    .replace(hfst.EPSILON, '')              # really necessary?
    
    logging.debug('input_str:  %s', input_str)

    try:
        complete_outputs = sw.window_size_1_2(
            input_str, None, None, model['flag_encoder'],
            gl_config['result_num'], model['composition'])
        
        for i, complete_output in enumerate(complete_outputs):
            if not gl_config['apply_lm']:
                complete_output.n_best(1)

            complete_output = sw.remove_flags(
                complete_output, model['flag_encoder'])
            if gl_config['apply_lm']:
                _apply_lm(complete_output)
            output_str = _output_tr_to_string(complete_output)
            _save_output_str(output_str, i)
            logging.debug('output_str: %s', output_str)

    except Exception as e:
        logging.exception('exception for window result of "%s"' % input_str)
        raise e
    
    return basename, input_str, output_str


def parallel_process(input_pairs, num_processes):
    with mp.Pool(processes=num_processes) as pool:
        result = pool.starmap_async(
            correct_string, input_pairs,
            error_callback=logging.error)
        result.wait()
        if result.successful():
            return result.get()
        else:
            raise RuntimeError('error during parallel processing')


def print_results(results, gt_dict):
    n = len(results)
    for i, (basename, input_str, output_str) in enumerate(results):
        print("%03d/%03d: %s" % (i+1, n, basename))
        print(input_str)
        print(output_str)
        print(gt_dict[basename])
        print()


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

    global model, gl_config
    
    # parse command-line arguments and set up various parameters
    args = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    gl_config = {
        'result_num' : args.result_num,
        'apply_lm' : args.apply_lm,
        'output_suffix' : args.output_suffix,
        'directory' : args.directory,
    }
    sw.REJECTION_WEIGHT = args.rejection_weight
    
    # load all transducers and build a model out of them
    model = prepare_model(
        args.punctuation,
        apply_lm = args.apply_lm,
        composition_depth = args.composition_depth,
        words_per_window = args.words_per_window,
        rejection_weight = args.rejection_weight,
        result_num = args.result_num)

    # load test data
    gt_dict = helper.create_dict(args.directory, 'gt.txt')
    ocr_dict = helper.create_dict(args.directory, args.input_suffix)

    # process test data and output results
    results = parallel_process(ocr_dict.items(), args.processes) \
              if args.processes > 1 \
              else [correct_string(basename, input_str) \
                    for basename, input_str in ocr_dict.items()]
    print_results(results, gt_dict)


if __name__ == '__main__':
    main()

