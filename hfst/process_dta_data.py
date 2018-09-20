from functools import reduce
import spacy
import helper


def create_lexicon(line_dict, nlp):
    """Create lexicon with frequencies from dict of lines. Words and punctation marks
    are inserted into separate dicts."""

    # TODO: Bindestriche behandeln. Momentan werden sie abgetrennt vor dem
    # Hinzufügen zum Lexikon. Man müsste halbe Worte weglassen und
    # zusammengesetzte Zeilen für die Erstellung des Lexikons nutzen.
    # TODO: Groß-/Kleinschreibung wie behandeln? Momentan wird jedes
    # Wort in der kleingeschriebenen und der großgeschriebene Variante
    # zum Lexikon hinzugefügt. Später vermutlich eher durch sowas wie
    # {CAP}?

    lexicon_dict = {}
    punctation_dict = {}
    open_bracket_dict = {}
    close_bracket_dict = {}

    lines = []

    if type(line_dict) is dict:
        lines = list(line_dict.values())
    elif type(line_dict) is list:
        lines = line_dict
    else:
        print('Creating lexicon failed: no dict or list given, but ', type(line_dict))

    #for line in lines[0:100]:
    for line in lines:

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
                if text.isdigit():
                    text = len(text) * '1'
                    lexicon_dict[text] = lexicon_dict.setdefault(text, 0) + 1

                else:

                    # add a word both uppercase and lowercase
                    lexicon_dict[text] = lexicon_dict.setdefault(text, 0) + 1
                    if token.text[0].isupper():
                        #print(text.lower())
                        lexicon_dict[text.lower()] = lexicon_dict.setdefault(text.lower(), 0) + 1
                    else:
                        if len(text) >= 2:
                            lexicon_dict[text[0].upper() + text[1:]] = lexicon_dict.setdefault(text[0].upper() + text[1:], 0) + 1
                        else:
                            lexicon_dict[text[0].upper()] = lexicon_dict.setdefault(text[0].upper(), 0) + 1

    return lexicon_dict, punctation_dict, open_bracket_dict, close_bracket_dict


def main():
    """Read text data, tokenize it, and count the words, separated into
    lexicon words, punctuation, and opening/closing brackets. Write lexica
    to txt files with word and (negative logarithm of) relative_frequency
    tab-separated."""

    # load dta19-reduced data
    #path = '../dta19-reduced/traindata/'
    path = '../dta19-reduced/testdata/'
    gt_dict = helper.create_dict(path, 'gt')

    # load dta-komplett
    #dta_file = '../Daten/ngram-model/gesamt_dta.txt'
    #with open(dta_file) as f:
    #    gt_dict = list(f)

    # read dta19-reduced data
    ##frak3_dict = create_dict(path, 'deu-frak3')
    #fraktur4_dict = create_dict(path, 'Fraktur4')
    ##foo4_dict = create_dict(path, 'foo4')

    # get dicts containing a lexicon of words, punctuation, opening/closing
    # brackets
    nlp = spacy.load('de')
    lexicon_dict, punctuation_dict, open_bracket_dict, close_bracket_dict = create_lexicon(gt_dict, nlp)

    #line_id = '05110'

    #print(gt_dict[line_id])
    #print(frak3_dict[line_id])
    #print(fraktur4_dict[line_id])
    #print(foo4_dict[line_id])

    # get relative frequency of absolute counts
    for dic in lexicon_dict, punctuation_dict, open_bracket_dict, close_bracket_dict:
        dic = helper.convert_to_relative_freq(dic)

    #print(lexicon_dict['der'])
    #print(punctation_dict[','])

    # write dicts to txt files: word tab relative_frequency
    helper.write_lexicon(lexicon_dict, 'dta_lexicon.txt')
    helper.write_lexicon(punctuation_dict, 'dta_punctuation.txt')
    helper.write_lexicon(open_bracket_dict, 'open_bracket.txt')
    helper.write_lexicon(close_bracket_dict, 'close_bracket.txt')

if __name__ == '__main__':
    main()



