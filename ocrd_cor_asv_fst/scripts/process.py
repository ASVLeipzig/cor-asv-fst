from os import listdir
import os.path

import argparse
import hfst
import logging
import tempfile
import multiprocessing as mp
import pynini

from ..lib.extensions.composition import pyComposition
from ..lib.sliding_window import \
    lexicon_to_window_fst, lattice_shortest_path, process_string
from ..lib.helper import \
    save_transducer, load_transducer, load_pairs_from_file, \
    load_pairs_from_dir, save_pairs_to_file, save_pairs_to_dir

# globals (for painless cow-semantic shared memory fork-based multiprocessing)
model = {}
gl_config = {}
    

def prepare_composition(lexicon_transducer, error_transducer, result_num, rejection_weight):
    # write lexicon and error transducer files in OpenFST format
    # (cannot use one file for both with OpenFST::Read)
    result = None
    with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-error') as error_f:
        with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-lexicon') as lexicon_f:
            save_transducer(
                error_f.name, error_transducer, hfst_format=False,
                type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            save_transducer(
                lexicon_f.name, lexicon_transducer, hfst_format=False,
                type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            
            result = (pynini.arcsort(pynini.Fst.read(error_f.name)),
                      pynini.arcsort(pynini.Fst.read(lexicon_f.name)))
            # result = pyComposition(
            #     error_f.name, lexicon_f.name,
            #     result_num, rejection_weight)
    return result


def prepare_model(lexicon_file, error_model_file, **kwargs):
    # lexicon_fst = load_transducer(lexicon_file)
    # error_fst = load_transducer(error_model_file)
    # window_fst = lexicon_to_window_fst(\
    #     lexicon_fst, kwargs['words_per_window'])
    # result = prepare_composition(\
    #     window_fst, error_fst, kwargs['result_num'], kwargs['rejection_weight'])
    lexicon_fst = pynini.Fst.read(lexicon_file)
    window_fst = lexicon_to_window_fst(lexicon_fst, kwargs['words_per_window'])
    window_fst.arcsort()
    error_fst = pynini.Fst.read(error_model_file)
    result = (error_fst, window_fst)
    return result


# needs to be global for multiprocessing
def correct_string(basename, input_str):
    global model, gl_config
    logging.debug('input_str:  %s', input_str)
    lattice = process_string(
        input_str, model,
        rejection_weight=gl_config['rejection_weight'])
    output_str = lattice_shortest_path(lattice)
    logging.debug('output_str: %s', output_str)
    return basename, output_str


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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction ocrd-cor-asv-fst batch-processor '
                    'tool')
    # TODO HERE
    parser.add_argument(
        '-l', '--lexicon-file', metavar='FILE', type=str, default=None,
        help='file containing the lexicon transducer')
    parser.add_argument(
        '-e', '--error-model-file', metavar='FILE', type=str, default=None,
        help='file containing the error model transducer')
    parser.add_argument(
        '-d', '--directory', metavar='PATH', default=None,
        help='directory for input and output files')
    parser.add_argument(
        '-I', '--input-suffix', metavar='SUF', type=str, default=None,
        help='input (OCR) filenames suffix')
    parser.add_argument(
        '-i', '--input-file', metavar='FILE', type=str, default=None,
        help='file containing the input data in two-column format')
    parser.add_argument(
        '-O', '--output-suffix', metavar='SUF', type=str, default=None,
        help='output (corrected) filenames suffix')
    parser.add_argument(
        '-o', '--output-file', metavar='FILE', type=str, default=None,
        help='file to write the output data in two-column format')
    parser.add_argument(
        '-W', '--words-per-window', metavar='NUM', type=int, default=3,
        help='maximum number of words in one window')
    parser.add_argument(
        '-R', '--result-num', metavar='NUM', type=int, default=10,
        help='result paths per window')
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
    Read OCR-ed lines:
    - either from files following the path scheme <directory>/<ID>.<suffix>,
      where each file contains one line of text,
    - or from a single, two-column file: <ID> <TAB> <line>.
    Correct each line and save output according to one of the two
    above-mentioned schemata.
    """

    global model, gl_config
    
    # parse command-line arguments and set up various parameters
    args = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    gl_config = {
        'result_num' : args.result_num,
        'apply_lm' : args.apply_lm,
        'output_suffix' : args.output_suffix,
        'rejection_weight' : args.rejection_weight,
    }

    # check the validity of parameters specifying input/output
    if args.input_file is None and \
            (args.input_suffix is None or args.directory is None):
        raise RuntimeError('No input data supplied! You have to specify either'
                           ' -i or -I and the data directory.')
    if args.output_file is None and \
            (args.output_suffix is None or args.directory is None):
        raise RuntimeError('No output file speficied! You have to specify '
                           'either -o or -O and the data directory.')

    
    # load all transducers and build a model out of them
    model = prepare_model(
        args.lexicon_file,
        args.error_model_file,
        apply_lm = args.apply_lm,
        words_per_window = args.words_per_window,
        rejection_weight = args.rejection_weight,
        result_num = args.result_num)

    # load input data
    pairs = load_pairs_from_file(args.input_file) \
            if args.input_file is not None \
            else load_pairs_from_dir(args.directory, args.input_suffix)

    # process
    results = parallel_process(pairs, args.processes) \
              if args.processes > 1 \
              else [correct_string(basename, input_str) \
                    for basename, input_str in pairs]

    # save results
    if args.output_file is not None:
        save_pairs_to_file(results, args.output_file)
    else:
        save_pairs_to_dir(results, args.directory, args.output_suffix)


if __name__ == '__main__':
    main()

