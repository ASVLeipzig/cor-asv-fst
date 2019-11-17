import argparse
import logging
import multiprocessing as mp
import pynini

from ocrd_keraslm.lib import Rater

from ..lib.latticegen import FSTLatticeGenerator, lattice_shortest_path
from ..lib.helper import \
    load_pairs_from_file, load_pairs_from_dir, \
    save_pairs_to_file, save_pairs_to_dir


# globals (for painless cow-semantic shared memory fork-based multiprocessing)
PROCESSOR = None


class PlaintextProcessor:
    '''
    Class responsible for complete processing of plaintext input:
    - lattice generation using lib.latticegen.FSTLatticeGenerator,
    - rescoring with a language model, if applicable,
    - best path search in the lattice.
    '''

    def __init__(self, latticegen, lm):
        self.latticegen = latticegen
        self.lm = lm

    def correct_string(self, input_str):
        logging.debug('input_str:  %s', input_str)
        lattice = self.latticegen.lattice_from_string(input_str)
        output_str = None
        if self.lm is not None:
            path = self._lm_find_best_path(lattice)
            output_str = ' '.join(p[1].Unicode for p in path)
        else:
            output_str = lattice_shortest_path(lattice)
        logging.debug('output_str: %s', output_str)
        return output_str

    def _lm_find_best_path(self, lattice):
        path, entropy, traceback = self.lm.rate_best(
            lattice, 0, max(lattice.nodes()),
            start_traceback = None,
            context = [0],
            lm_weight = 1,
            beam_width = 3,
            beam_clustering_dist = 5)
        path, entropy, traceback = \
            self.lm.next_path(traceback[0], ([], traceback[1]))
        return path


# needs to be global for multiprocessing
def correct_string(basename, input_str):
    global PROCESSOR
    return basename, PROCESSOR.correct_string(input_str)


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
    parser.add_argument(
        '-l', '--lexicon-file', metavar='FILE', type=str, default=None,
        help='file containing the lexicon transducer')
    parser.add_argument(
        '-e', '--error-model-file', metavar='FILE', type=str, default=None,
        help='file containing the error model transducer')
    parser.add_argument(
        '-m', '--language-model-file', metavar='FILE', type=str, default=None,
        help='file containing the language model')
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
        '-P', '--pruning-weight', metavar='WEIGHT', type=float, default=5,
        help='transition weight for pruning the hypotheses space')
    parser.add_argument(
        '-J', '--rejection-weight', metavar='WEIGHT', type=float, default=1.5,
        help='transition weight (per character) for unchanged input window')
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
    '''
    Read OCR-ed lines:
    - either from files following the path scheme <directory>/<ID>.<suffix>,
      where each file contains one line of text,
    - or from a single, two-column file: <ID> <TAB> <line>.
    Correct each line and save output according to one of the two
    above-mentioned schemata.
    '''

    global PROCESSOR
    
    # parse command-line arguments and set up various parameters
    args = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(args.log_level))

    # check the validity of parameters specifying input/output
    if args.input_file is None and \
            (args.input_suffix is None or args.directory is None):
        raise RuntimeError('No input data supplied! You have to specify either'
                           ' -i or -I and the data directory.')
    if args.output_file is None and \
            (args.output_suffix is None or args.directory is None):
        raise RuntimeError('No output file speficied! You have to specify '
                           'either -o or -O and the data directory.')

    using_lm = (args.language_model_file is not None)
    latticegen = FSTLatticeGenerator(
        args.lexicon_file,
        args.error_model_file,
        lattice_format   = 'networkx' if using_lm else 'fst',
        words_per_window = args.words_per_window,
        rejection_weight = args.rejection_weight,
        pruning_weight   = args.pruning_weight)
    lm = None
    if using_lm:
        lm = Rater(logger=logging)
        lm.load_config(args.language_model_file)
        # overrides for incremental mode necessary before compilation:
        lm.stateful = False         # no implicit state transfer
        lm.incremental = True       # but explicit state transfer
        lm.configure()
        lm.load_weights(args.language_model_file)
    PROCESSOR = PlaintextProcessor(latticegen, lm)

    # load input data
    pairs = load_pairs_from_file(args.input_file) \
            if args.input_file is not None \
            else load_pairs_from_dir(args.directory, args.input_suffix)

    # process
    results = parallel_process(pairs, args.processes) \
              if args.processes > 1 \
              else [(basename, PROCESSOR.correct_string(input_str)) \
                    for basename, input_str in pairs]

    # save results
    if args.output_file is not None:
        save_pairs_to_file(results, args.output_file)
    else:
        save_pairs_to_dir(results, args.directory, args.output_suffix)


if __name__ == '__main__':
    main()

