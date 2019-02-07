import argparse
from collections import defaultdict, namedtuple
from functools import reduce
import hfst
from operator import itemgetter
import re

# to install models, do: `python -m spacy download de` after installation
import spacy
import spacy.tokenizer

import helper


MIN_LINE_LENGTH = 3
OPENING_BRACKETS = ['"', '»', '(', '„']
CLOSING_BRACKETS = ['"', '«', ')', '“', '‘', "'"]
UMLAUTS = { 'ä': 'a\u0364', 'ö': 'o\u0364', 'ü': 'u\u0364', 'Ä': 'A\u0364',
            'Ö': 'O\u0364', 'Ü': 'U\u0364'}

Lexicon = namedtuple(
    'Lexicon',
     ['opening_brackets', 'closing_brackets', 'punctuation', 'words'])


def get_digit_tuples():
    """Gives tuple of all pairs of identical numbers.
    This is used to replace the ('1', '1') transitions in the lexicon by
    all possible numbers."""

    return tuple([(str(i), str(i)) for i in range(10)])


def setup_spacy(use_gpu=False):
    if use_gpu:
        spacy.require_gpu()
        spacy.util.use_gpu(0)
    # disable everything we don't have at runtime either
    nlp = spacy.load('de', disable=['parser', 'ner'])
    infix_re = spacy.util.compile_infix_regex(
        nlp.Defaults.infixes +
        ['—',                   # numeric dash: (?<=[0-9])—(?=[0-9])
         '/'])                  # maybe more restrictive?
    suffix_re = spacy.util.compile_suffix_regex(
        nlp.Defaults.suffixes +
        ('/',)) # maybe more restrictive?
    # '〟' as historic quotation mark (left and right)
    # '〃' as historic quotation mark (at the start of the line!)
    # '‟' as historic quotation mark (at the start of the line!)
    # '›' and '‹' as historic quotation marks (maybe goes away with NFC?)
    # '⟨' and '⟩' parentheses (maybe goes away with NFC?)
    # '⁽' and '⁾' parentheses (maybe goes away with NFC?)
    # '〈' and '〉' brackets (maybe goes away with NFC?)
    # '‹' and '›' as historic quotation mark
    # '’' as historic apostrophe
    # '—' as dash, even when written like a prefix
    # \u+feff (byte order mark) as prefix

    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab,
        token_match=nlp.tokenizer.token_match,
        prefix_search=nlp.tokenizer.prefix_search,
        suffix_search=nlp.tokenizer.suffix_search,
        infix_finditer=infix_re.finditer)
    return nlp


def build_lexicon(lines):
    """Create lexicon with frequencies from dict of lines. Words and
    punctation marks are inserted into separate dicts."""

    # TODO: Bindestriche behandeln. Momentan werden sie abgetrennt vor dem
    # Hinzufügen zum Lexikon. Man müsste halbe Worte weglassen und
    # zusammengesetzte Zeilen für die Erstellung des Lexikons nutzen.
    # TODO: Groß-/Kleinschreibung wie behandeln? Momentan wird jedes
    # Wort in der kleingeschriebenen und der großgeschriebene Variante
    # zum Lexikon hinzugefügt (mit gleicher Häufigkeit).
    # Später vermutlich eher durch sowas wie {CAP}?

    def _is_opening_bracket(token):
        return token.text in OPENING_BRACKETS

    def _is_closing_bracket(token):
        return token.text in CLOSING_BRACKETS

    def _is_punctuation(token):
        # punctuation marks must not contain letters or numbers
        # hyphens in the middle of the text are treated as words
        return token.pos_ == 'PUNCT' and token.text != '—' and \
            not any(c.isalpha() or c.isnumeric() for c in token.text)

    def _handle_problematic_cases(token):
        if token.text.strip() != token.text:
            logging.warning('Token contains leading or trailing '
                            'whitespaces: \'{}\''.format(token.text))
        if len(token.text) > 1 and token.text.endswith('—'):
            logging.warning('Possible tokenization error: \'{}\''
                            .format(token.text))

    lexicon = Lexicon(
        opening_brackets=defaultdict(lambda: 0),
        closing_brackets=defaultdict(lambda: 0),
        punctuation=defaultdict(lambda: 0),
        words=defaultdict(lambda: 0))
    umlauttrans = str.maketrans(UMLAUTS)
    # '−' as sign prefix
    # '√' as prefix?
    # ¹²³⁴⁵⁶⁷⁸⁹⁰ digits (maybe goes away with NFC?)
    num_re = re.compile('[0-9]{1,3}([,.]?[0-9]{3})*([.,][0-9]*)?')
    nlp = setup_spacy()
    
    if isinstance(lines, dict):
        lines = lines.values()
    elif hasattr(lines, '__iter__'): # accept generators
        lines = map(itemgetter(1), lines)
    else:
        raise RuntimeError('Creating lexicon failed: %s given, but dict or '
                           'list expected' % type(lines))

    for line in lines:
        if len(line) < MIN_LINE_LENGTH:
            continue
        for token in nlp(line):
            _handle_problematic_cases(token)
            if _is_opening_bracket(token):
                lexicon.opening_brackets[token.text] += 1
            elif _is_closing_bracket(token):
                lexicon.closing_brackets[token.text] += 1
            elif _is_punctuation(token):
                lexicon.punctuation[token.text] += 1
            else:
                text = token.text.translate(umlauttrans)
                if text.isdigit() or num_re.match(text):
                    text = len(text) * '1'
                lexicon.words[text] += 1
                # include also the (un)capitalized variant
                recap = text.lower() \
                        if text[0].isupper() \
                        else text.capitalize()
                if recap != text:
                    lexicon.words[recap] += 1

    return lexicon


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='OCR post-correction ocrd-cor-asv-fst lexicon extractor')
    parser.add_argument(
        'directory', metavar='PATH', help='directory for input files')
    parser.add_argument(
        '-G', '--gt-suffix', metavar='SUF', type=str, default='gt.txt',
        help='clean (Ground Truth) filenames suffix') # useful for DTA (.tcf.txt)
    parser.add_argument(
        '-U', '--use-gpu', action='store_true', default=False,
        help='use GPU instead of CPU for tokenization (spacy)')
    return parser.parse_args()


def main():
    '''
    Read text data, tokenize it, and count the words, separated into
    lexicon words, punctuation, and opening/closing brackets. Convert
    frequencies to minus log-relative frequencies and compile transducers from
    the resulting data.
    '''

    args = parse_arguments()

    # load dta19-reduced data
    #path = '../dta19-reduced/traindata/'
    #path = '../dta19-reduced/testdata/'
    gt_filenames = helper.get_filenames(args.directory, args.gt_suffix)
    gt_data = helper.generate_content(args.directory, gt_filenames)

    # load dta-komplett
    #dta_file = '../Daten/ngram-model/gesamt_dta.txt'
    #with open(dta_file) as f:
    #    gt_dict = list(f)

    # read dta19-reduced data
    ##frak3_dict = create_dict(path, 'deu-frak3')
    #fraktur4_dict = create_dict(path, 'Fraktur4')
    ##foo4_dict = create_dict(path, 'foo4')

    lexicon = build_lexicon(gt_data)

    #line_id = '05110'

    #print(gt_dict[line_id])
    #print(frak3_dict[line_id])
    #print(fraktur4_dict[line_id])
    #print(foo4_dict[line_id])

    lexicon_transducer = helper.transducer_from_dict(
        helper.convert_to_log_relative_freq(lexicon.words))
    punctuation_transducer = helper.transducer_from_dict(
        helper.convert_to_log_relative_freq(lexicon.punctuation))
    open_bracket_transducer = helper.transducer_from_dict(
        helper.convert_to_log_relative_freq(lexicon.opening_brackets))
    close_bracket_transducer = helper.transducer_from_dict(
        helper.convert_to_log_relative_freq(lexicon.closing_brackets))

    # in the lexicon dict, numbers are counted as sequences of 1
    # thus, they are replaced by any possible number of the according length
    lexicon_transducer.substitute(('1', '1'), get_digit_tuples())

    helper.save_transducer('lexicon_transducer_dta.hfst', lexicon_transducer)
    helper.save_transducer('punctuation_transducer_dta.hfst', punctuation_transducer)
    helper.save_transducer('open_bracket_transducer_dta.hfst', open_bracket_transducer)
    helper.save_transducer('close_bracket_transducer_dta.hfst', close_bracket_transducer)


if __name__ == '__main__':
    main()
