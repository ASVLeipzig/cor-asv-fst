from functools import reduce
import argparse
import re

# to install models, do: `python -m spacy download de` after installation
import spacy
import spacy.tokenizer

import helper


def create_lexicon(lines, nlp):
    """Create lexicon with frequencies from dict of lines. Words and
       punctation marks are inserted into separate dicts."""

    # TODO: Bindestriche behandeln. Momentan werden sie abgetrennt vor dem
    # Hinzufügen zum Lexikon. Man müsste halbe Worte weglassen und
    # zusammengesetzte Zeilen für die Erstellung des Lexikons nutzen.
    # TODO: Groß-/Kleinschreibung wie behandeln? Momentan wird jedes
    # Wort in der kleingeschriebenen und der großgeschriebene Variante
    # zum Lexikon hinzugefügt (mit gleicher Häufigkeit).
    # Später vermutlich eher durch sowas wie {CAP}?

    lexicon_dict = {}
    punctation_dict = {}
    open_bracket_dict = {}
    close_bracket_dict = {}
    umlautset = set("äöüÄÖÜ")
    umlauttrans = str.maketrans({'ä': 'aͤ', 'ö': 'oͤ', 'ü': 'uͤ', 'Ä': 'Aͤ', 'Ö': 'Oͤ', 'Ü': 'Uͤ'})
    num_re = re.compile('[0-9]{1,3}([,.]?[0-9]{3})*([.,][0-9]*)?')
    # '−' as sign prefix
    # '√' as prefix?
    # ¹²³⁴⁵⁶⁷⁸⁹⁰ digits (maybe goes away with NFC?)
    
    if type(lines) is dict:
        lines = lines.values()
    elif hasattr(lines, '__iter__'): # accept generators
        lines = map(lambda x: x[1], lines)
    else:
        raise Exception('Creating lexicon failed: %s given, but dict or list expected' % type(lines))

    #for line in lines[0:100]:
    for line in lines:
        #print(line)

        if len(line) < 3:
            continue

        # tokenize line
        doc = nlp(line)

        for token in doc:
            #print(token.text, '\t', token.lemma_, '\t', token.pos_,
            #token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

            text = token.text.strip()

            if len(text) == 0:
                continue

            # if word ends with hyphen, add hyphen as a punctuation mark
            if len(text) > 1 and text[-1] == '—':
                text = text[0:-1]
                punctation_dict['—'] = punctation_dict.setdefault('—', 0) + 1

            # handle open bracket marks
            if text in ['"', '»', '(', '„']:
                open_bracket_dict[text] = open_bracket_dict.setdefault(text, 0) + 1

            # handle close bracket marks
            elif text in ['"', '«', ')', '“', '‘', "'"]:
                close_bracket_dict[text] = close_bracket_dict.setdefault(text, 0) + 1

            # punctuation marks must not contain letters or numbers
            # hyphens in the middle of the text are treated as words
            elif token.pos_ == 'PUNCT' and text != '—' and \
                not reduce(lambda x,y: x or y, list(map(lambda x: x.isalpha()\
                or x.isnumeric(), text)), False):
                punctation_dict[text] = punctation_dict.setdefault(text, 0) + 1

            else:

                # numbers are normalized to 1 to be replaced by a number
                # transducer later, but the length of a number is preserved
                if text.isdigit() or num_re.match(text):
                    text = len(text) * '1'
                    lexicon_dict[text] = lexicon_dict.setdefault(text, 0) + 1

                else:
                    # normalize umlauts to decomposed form (but precomposed variant will be allowed at runtime too):
                    if umlautset.intersection(set(text)):
                        text = text.translate(umlauttrans)
                    # add a word both uppercase and lowercase
                    # FIXME: instead gather statistics separately, then keep only most frequent form (but allow titlecase after sentence punctuation or beginning of line at runtime too)
                    lexicon_dict[text] = lexicon_dict.setdefault(text, 0) + 1
                    # despite Python #6412, str.istitle() and str.title() are still buggy with combining diacritics # if token.text.istitle():
                    if token.text[0].isupper():
                        recap = text.lower()
                        lexicon_dict[recap] = lexicon_dict.setdefault(recap, 0) + 1
                    else:
                        # despite Python #6412, str.istitle() and str.title() are still buggy with combining diacritics # recap = text.title()
                        recap = text.capitalize()
                        lexicon_dict[recap] = lexicon_dict.setdefault(recap, 0) + 1

    return lexicon_dict, punctation_dict, open_bracket_dict, close_bracket_dict

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
    """Read text data, tokenize it, and count the words, separated into
    lexicon words, punctuation, and opening/closing brackets. Write lexica
    to txt files with word and (negative logarithm of) relative_frequency
    tab-separated."""

    args = parse_arguments()

    # load dta19-reduced data
    #path = '../dta19-reduced/traindata/'
    #path = '../dta19-reduced/testdata/'
    gt_filenames = helper.get_filenames(args.directory + "/", args.gt_suffix)
    gt_data = helper.generate_content(args.directory + "/", gt_filenames)

    # load dta-komplett
    #dta_file = '../Daten/ngram-model/gesamt_dta.txt'
    #with open(dta_file) as f:
    #    gt_dict = list(f)

    # read dta19-reduced data
    ##frak3_dict = create_dict(path, 'deu-frak3')
    #fraktur4_dict = create_dict(path, 'Fraktur4')
    ##foo4_dict = create_dict(path, 'foo4')

    nlp = setup_spacy(args.use_gpu)
    lexicon_dict, punctuation_dict, open_bracket_dict, close_bracket_dict = \
        create_lexicon(gt_data, nlp)

    #line_id = '05110'

    #print(gt_dict[line_id])
    #print(frak3_dict[line_id])
    #print(fraktur4_dict[line_id])
    #print(foo4_dict[line_id])

    # get relative frequency of absolute counts
    # write dicts to txt files: word tab relative_frequency
    lexicon_dict = helper.convert_to_relative_freq(lexicon_dict)
    helper.write_lexicon(lexicon_dict, 'dta_lexicon.txt')
    punctuation_dict = helper.convert_to_relative_freq(punctuation_dict)
    helper.write_lexicon(punctuation_dict, 'dta_punctuation.txt')
    open_bracket_dict = helper.convert_to_relative_freq(open_bracket_dict)
    helper.write_lexicon(open_bracket_dict, 'open_bracket.txt')
    close_bracket_dict = helper.convert_to_relative_freq(close_bracket_dict)
    helper.write_lexicon(close_bracket_dict, 'close_bracket.txt')

if __name__ == '__main__':
    main()

