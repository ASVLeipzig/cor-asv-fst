import argparse
import logging
import numpy as np
from operator import itemgetter

from ..lib.lexicon import build_lexicon, lexicon_to_fst
from ..lib.error_simp import \
    get_confusion_dicts, compile_single_error_transducer, \
    combine_error_transducers
from ..lib.helper import \
    load_pairs_from_file, load_pairs_from_dir, load_lines_from_file, \
    load_wordlist_from_file
from ..lib.error_st import \
    fit, compile_transducer, load_ngrams, matrix_to_mappings, save_ngrams, \
    training_pairs_to_ngrams


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction model training')
    # GENERAL PARAMETERS
    parser.add_argument(
        '-l', '--lexicon-file', metavar='FILE', type=str, default=None,
        help='file to save the trained lexicon')
    parser.add_argument(
        '-e', '--error-model-file', metavar='FILE', type=str, default=None,
        help='file to save the trained error model')
    parser.add_argument(
        '-t', '--training-file', metavar='FILE', type=str, default=None,
        help='file containing training data in two-column format (OCR, GT)')
    parser.add_argument(
        '-L', '--log-level', metavar='LEVEL', type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='verbosity of logging output (standard log levels)')
    # alternative method of passing training data - two files:
    parser.add_argument(
        '-i', '--input-file', metavar='FILE', type=str, default=None,
        help='file containing the input data in two-column format')
    parser.add_argument(
        '-g', '--gt-file', metavar='FILE', type=str, default=None,
        help='file containing the ground truth data in two-column format')
    # yet alternative method of passing training data -- multiple files:
    parser.add_argument(
        '-I', '--input-suffix', metavar='SUF', type=str, default=None,
        help='input (OCR) filenames suffix')
    parser.add_argument(
        '-G', '--gt-suffix', metavar='SUF', type=str, default=None,
        help='ground truth filenames suffix')
    parser.add_argument(
        '-d', '--directory', metavar='PATH', default=None,
        help='directory for training files')
    # PARAMETERS FOR TRAINING THE LEXICON
    parser.add_argument(
        '-c', '--corpus-file', metavar='FILE', type=str, default=None,
        help='a file containing a plaintext corpus')
    parser.add_argument(
        '-w', '--wordlist-file', metavar='FILE', type=str, default=None,
        help='a file containing a wordlist in two-column format '
             '(word <tab> frequency)')
    parser.add_argument(
        '-P', '--punctuation', metavar='MODEL', type=str,
        choices=['bracket', 'lm', 'preserve'], default='bracket',
        help='how to model punctuation between words (bracketing rules, '
             'inter-word language model, or keep unchanged)')
    parser.add_argument(
        '-D', '--composition-depth', metavar='NUM', type=int, default=2,
        help='max. number of lexicon words that can be concatenated')
    parser.add_argument(
        '--lexicon-added-word-cost', metavar='NUM', type=float, default=0,
        help='a constant to add to the weights of every word in lexicon')
    parser.add_argument(
        '--unweighted-lexicon', action='store_true', default=False,
        help='train an unweighted lexicon (use for combining with a language '
             'model)')
    # PARAMETERS FOR TRAINING THE ERROR MODEL
    parser.add_argument(
        '-T', '--error-model-type', metavar='MODEL', type=str,
        choices=['simple', 'st'], default='simple',
        help='type of the error model')
    parser.add_argument(
        '-p', '--preserve-punctuation', action='store_true', default=False,
        help='ignore edits to/from non-alphanumeric or non-space characters '
             '(only the \'simple\' model)')
    parser.add_argument(
        '--min-context', metavar='NUM', type=int, default=1,
        help='minimum size of context count edits at')
    parser.add_argument(
        '-C', '--max-context', metavar='NUM', type=int, default=3,
        help='maximum size of context count edits at')
    parser.add_argument(
        '-E', '--max-errors', metavar='NUM', type=int, default=3,
        help='maximum number of errors the resulting FST can correct '
             '(applicable within one window, i.e. a certain number of words)')
    # only ST error model:
    parser.add_argument(
        '-N', '--max-ngrams', metavar='NUM', type=int, default=1000,
        help='max. number of n-grams used in ST error model training')
    parser.add_argument(
        '-W', '--weight-threshold', metavar='NUM', type=float, default=5.0,
        help='max. cost of transformations included in the error model')
    parser.add_argument(
        '--crossentr-threshold', metavar='NUM', type=float, default=0.001,
        help='threshold on cross-entropy for stopping ST error model training')
    parser.add_argument(
        '--ngrams-file', metavar='FILE', type=str,
        help='')
    parser.add_argument(
        '--weights-file', metavar='FILE', type=str,
        help='')
    parser.add_argument(
        '--load-weights-from', metavar='FILE', type=str,
        help='')
    return parser.parse_args()


def main():

    def _load_training_pairs(args):
        if args.training_file is not None:
            return load_pairs_from_file(args.training_file)
        elif args.input_suffix is not None \
                and args.gt_suffix is not None \
                and args.directory is not None:
            ocr_dict = dict(load_pairs_from_dir(\
                args.directory, args.input_suffix))
            gt_dict = dict(load_pairs_from_dir(\
                args.directory, args.gt_suffix))
            return [(ocr_dict[key], gt_dict[key]) \
                    for key in set(ocr_dict) & set(gt_dict)]
        else:
            return []

    def _load_lexicon_training_data(args):
        training_dict = None
        training_pairs = _load_training_pairs(args)
        training_lines = list(map(itemgetter(1), training_pairs))
        if args.corpus_file is not None:
            training_lines.extend(load_lines_from_file(args.corpus_file))
        if args.wordlist_file is not None:
            training_dict = load_wordlist_from_file(args.wordlist_file)
        if not training_lines and not training_dict:
            logging.error('No training data supplied!')
        return training_lines, training_dict

    def _train_lexicon(args):
        training_lines, training_dict = _load_lexicon_training_data(args)
        lexicon = build_lexicon(training_lines, training_dict)
        tr = lexicon_to_fst(\
            lexicon, punctuation=args.punctuation,
            added_word_cost=args.lexicon_added_word_cost,
            unweighted=args.unweighted_lexicon)
        tr.write(args.lexicon_file)

    def _train_simple_error_model(args):
        training_pairs = _load_training_pairs(args)
        # FIXME this is silly, instead refactor the simple error model training
        # so that it accepts input in form of line pairs
        ocr_dict, gt_dict = {}, {}
        for i, (ocr_line, gt_line) in enumerate(training_pairs):
            ocr_dict[i] = ocr_line
            gt_dict[i] = gt_line

        confusion_dicts = get_confusion_dicts(\
            gt_dict, ocr_dict, args.max_context)
        single_error_transducers = \
            [compile_single_error_transducer(
                confusion_dicts[i],
                preserve_punct=args.preserve_punctuation) \
             for i in range(1, args.max_context+1)]
        combined_tr_dicts = combine_error_transducers(
            single_error_transducers,
            args.max_context,
            args.max_errors)
        # FIXME combine_error_transducers() should return a single FST instead
        # of this complicated dict structure
        target_context = \
            ''.join(map(str, range(args.min_context, args.max_context+1)))
        for tr_dict in combined_tr_dicts:
            if tr_dict['max_error'] == args.max_errors and \
                    tr_dict['context'] == target_context:
                # save_transducer(args.error_model_file, tr_dict['transducer'])
                tr_dict['transducer'].write(args.error_model_file)

    def _train_st_error_model(args):
        # FIXME implement
        # if weight file given -> load weights from there, otherwise train them
        ngrams, probs, ngr_probs = None, None, None
        if args.load_weights_from is not None:
            ngrams = load_ngrams(args.ngrams_file)
            with np.load(args.load_weights_from) as data:
                probs, ngr_probs = data['probs'], data['ngr_probs']
        else:
            training_pairs = _load_training_pairs(args)
            ngr_training_pairs, ngrams = training_pairs_to_ngrams(
                training_pairs,
                max_n=args.max_context, max_ngrams=args.max_ngrams)
            if args.ngrams_file is not None:
                save_ngrams(args.ngrams_file, ngrams)
            probs, ngr_probs = fit(
                ngr_training_pairs, ngrams,
                threshold=args.crossentr_threshold)
            if args.weights_file is not None:
                np.savez(args.weights_file, probs=probs, ngr_probs=ngr_probs)

        mappings = matrix_to_mappings(
            probs, ngrams, weight_threshold=args.weight_threshold)
        for input_str, output_str, weight in mappings:
            print('\''+input_str+'\'', '\''+output_str+'\'', weight, sep='\t')
        tr = compile_transducer(
               mappings, ngr_probs, max_errors=args.max_errors,
               max_context=args.max_context, weight_threshold=args.weight_threshold)
        tr.write(args.error_model_file)

    args = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(args.log_level))
    if args.lexicon_file is not None:
        _train_lexicon(args)
    else:
        logging.info('Skipping lexicon training.')
    if args.error_model_file is not None:
        if args.error_model_type == 'simple':
            _train_simple_error_model(args)
        elif args.error_model_type == 'st':
            _train_st_error_model(args)
    else:
        logging.info('Skipping error model training.')


if __name__ == '__main__':
    main()

