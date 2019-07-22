from collections import defaultdict, namedtuple
import logging
import re

# to install models, do: `python -m spacy download de` after installation
import spacy
import spacy.tokenizer

from .helper import \
    convert_to_log_relative_freq, transducer_from_dict


MIN_LINE_LENGTH = 3
OPENING_BRACKETS = ['"', '»', '(', '„']
CLOSING_BRACKETS = ['"', '«', ')', '“', '‘', "'"]
UMLAUTS = { 'ä': 'a\u0364', 'ö': 'o\u0364', 'ü': 'u\u0364', 'Ä': 'A\u0364',
            'Ö': 'O\u0364', 'Ü': 'U\u0364'}

Lexicon = namedtuple(
    'Lexicon',
     ['opening_brackets', 'closing_brackets', 'punctuation', 'words'])


def get_digit_tuples():
    '''
    Gives tuple of all pairs of identical numbers. This is used to
    replace the ('1', '1') transitions in the lexicon by all possible
    numbers.
    '''
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


def build_lexicon(lines, _dict = None):
    '''
    Create lexicon with frequencies from lines of plain text. Words and
    punctation marks are inserted into separate dicts.

    The additional parameter `_dict` is a dictionary: type -> frequency.
    If it is given, those types are additionally inserted into the
    lexicon as words (without any preprocessing).
    '''

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

    def _add_token_to_lexicon(token, freq = 1):
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
            lexicon.words[text] += freq
            # include also the (un)capitalized variant
            recap = text.lower() \
                    if text[0].isupper() \
                    else text.capitalize()
            if recap != text:
                lexicon.words[recap] += freq

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

    # process the text lines
    for line in lines:
        if len(line) < MIN_LINE_LENGTH:
            continue
        for token in nlp(line):
            _add_token_to_lexicon(token)

    # process the dictionary of words with frequencies
    if _dict is not None:
        for word, freq in _dict.items():
            lexicon.words[word] += freq

    return lexicon


def lexicon_to_fst(lexicon, punctuation='bracket', added_word_cost=0,
                   unweighted=False):
    words_dict = convert_to_log_relative_freq(lexicon.words)
    # add `added_word_cost` to the cost of every word
    # (FIXME this is a dirty workaround to reproduce the approximate behaviour
    # of the legacy implementation of the sliding window algorithm; it should
    # be replaced with something more theoretically sound)
    if added_word_cost != 0:
        logging.debug('adding {} to word costs'.format(added_word_cost))
        for w in words_dict:
            words_dict[w] += added_word_cost
    words_fst = transducer_from_dict(words_dict, unweighted=unweighted)
    punctuation_fst = transducer_from_dict(
        convert_to_log_relative_freq(lexicon.punctuation),
        unweighted=unweighted)
    open_bracket_fst = transducer_from_dict(
        convert_to_log_relative_freq(lexicon.opening_brackets),
        unweighted=unweighted)
    close_bracket_fst = transducer_from_dict(
        convert_to_log_relative_freq(lexicon.closing_brackets),
        unweighted=unweighted)

    # in the lexicon dict, numbers are counted as sequences of 1
    # thus, they are replaced by any possible number of the according length
    # FIXME restore converting digits to ones
    # words_fst.substitute(('1', '1'), get_digit_tuples())

    if punctuation == 'bracket':
        # TODO compounds
        result = open_bracket_fst.ques
        result.concat(words_fst)
        result.concat(punctuation_fst.ques)
        result.concat(close_bracket_fst.ques)

        # standardize the umlaut characters
        # FIXME restore the umlaut standardization
        # precompose_transducer = hfst.regex(
        #     '[a\u0364:ä|o\u0364:ö|u\u0364:ü|A\u0364:Ä|O\u0364:Ö|U\u0364:Ü|?]*')
        # result.compose(precompose_transducer)
        result.project(project_output=True)
        result.optimize(compute_props=True)
        result.push()
        return result
    else:
        # FIXME implement further punctuation methods
        raise NotImplementedError()

