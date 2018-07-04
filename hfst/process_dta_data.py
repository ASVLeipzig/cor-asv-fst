from os import listdir

from nltk import ngrams

from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = "ε"
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

from functools import reduce

import spacy

import math


def get_confusion_dict(gt_dict, raw_dict):

    corresponding_list = []     # list of tuples (gt_line, raw_line)
    difference_list = []        # list of tuples (gt_line, raw_line) with gt_line != raw_line

    for key in gt_dict.keys():
        raw_line = raw_dict.get(key, None)
        gt_line = gt_dict[key]
        if raw_line != None:
            corresponding = (gt_line, raw_line)
            corresponding_list.append(corresponding)
            if raw_line != gt_line:
                difference_list.append(corresponding)


    confusion_dict = [{}, {}, {}, {}]

    #for (gt_line, raw_line) in difference_list:
    for (gt_line, raw_line) in difference_list[1:100]:

        #print(gt_line)
        #print(raw_line)

        b = Sequence(gt_line)
        a = Sequence(raw_line)

        # Create a vocabulary and encode the sequences.
        v = Vocabulary()
        aEncoded = v.encodeSequence(a)
        bEncoded = v.encodeSequence(b)

        # Create a scoring and align the sequences using global aligner.
        scoring = SimpleScoring(2, -1)
        aligner = GlobalSequenceAligner(scoring, -2)
        score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)

        for encoded in encodeds:
            alignment = v.decodeSequenceAlignment(encoded)
            print(alignment)
            print('Alignment score:', alignment.score)
            print('Percent identity:', alignment.percentIdentity())

            if alignment.percentIdentity() < 100:

                firsts = ''.join(list(map(lambda x: x[0], alignment)))
                seconds = ''.join(list(map(lambda x: x[1], alignment)))

                for n in [1, 2, 3]:

                    grams_first = list(ngrams(firsts, n))
                    grams_second = list(ngrams(seconds, n))

                    for i, gram in enumerate(grams_first):
                        first = ''.join(gram)
                        second = ''.join(grams_second[i])
                        #print(first, second)

                        confusion_dict[n][first] = confusion_dict[n].setdefault(first, {})
                        confusion_dict[n][first][second] = confusion_dict[n][first].setdefault(second, 0) + 1


    #for i in [1, 2, 3]:
    #    print(confusion_dict[i].items())

    return(confusion_dict)




def get_txt_files(directory, model):
    return (f for f in listdir(directory) if f.endswith('.' + model + '.txt'))

def get_content(directory, file_generator):
    for x in file_generator:
        with open(directory + x) as f:
            yield [x.split('.')[0], f.read().strip()]

def create_dict(path, model):
    result_dict = {}
    files = get_txt_files(path, model)
    content = get_content(path, files)
    for file_id, line in content:
        result_dict[file_id] = line
    return result_dict




def create_lexicon(line_dict, nlp):
    """Create lexicon with frequencies from dict of lines. Words and punctation marks
    are inserted into separate dicts."""
    # TODO: Bindestriche behandeln. Momentan werden sie abgetrennt vor dem
    # Hinzufügen zum Lexikon. Man müsste halbe Worte weglassen und
    # zusammengesetzte Zeilen für die Erstellung des Lexikons nutzen.
    # TODO: Groß-/Kleinschreibung wie behandeln? Momentan wird jedes
    # Wort in der kleingeschriebenen und der großgeschriebene Variante
    # zum Lexikon hinzugefügt. Später vermutlich eher durch sowas wie
    # {CAP}.

    lexicon_dict = {}
    punctation_dict = {}

    lines = list(line_dict.values())

    #for line in lines[0:100]:
    for line in lines:

        # tokenize line
        doc = nlp(line)

        for token in doc:
            #print(token.text, '\t', token.lemma_, '\t', token.pos_,
            #token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

            # if word ends with hyphen, add hyphen as a punctuation mark
            text = token.text.strip()
            if len(text) > 1 and text[-1] == '—':
                text = text[0:-1]
                punctation_dict['—'] = punctation_dict.setdefault('—', 0) + 1

            # punctuation marks must not contain letters or numbers
            # hyphens in the middle of the text are treated as words
            if token.pos_ == 'PUNCT' and text != '—' and \
                reduce(lambda x,y: x or y, list(map(lambda x: x.isalpha() or x.isnumeric, text)), False):

                #print('PUNCT')
                #print(text)
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
                        lexicon_dict[text[0].upper() + text[1:]] = lexicon_dict.setdefault(text[0].upper() + text[1:], 0) + 1

    return lexicon_dict, punctation_dict


def convert_to_relative_freq(lexicon_dict):

    summed_freq = sum(list(lexicon_dict.values()))
    for key in list(lexicon_dict.keys()):
        lexicon_dict[key] = lexicon_dict[key] / summed_freq

    return lexicon_dict


def write_lexicon(lexicon_dict, filename, log=True):

    lexicon_list = list(lexicon_dict.items())

    with open(filename, 'w') as f:
        for word, freq in lexicon_list:
            if log:
                f.write(word + '\t' + str(- math.log(freq)) + '\n')
            else:
                f.write(word + '\t' + str(freq) + '\n')

    return


def main():

    path = '../dta19-reduced/traindata/'

    gt_dict = create_dict(path, 'gt')

    ##frak3_dict = create_dict(path, 'deu-frak3')
    #fraktur4_dict = create_dict(path, 'Fraktur4')
    ##foo4_dict = create_dict(path, 'foo4')

    #confusion_dict = get_confusion_dict(gt_dict, fraktur4_dict)

    nlp = spacy.load('de')
    lexicon_dict, punctation_dict = create_lexicon(gt_dict, nlp)

    #line_id = '05110'

    #print(gt_dict[line_id])
    #print(frak3_dict[line_id])
    #print(fraktur4_dict[line_id])
    #print(foo4_dict[line_id])

    for entry in punctation_dict.items():
        print(entry)
    for entry in lexicon_dict.items():
        print(entry)

    #print(lexicon_dict['der'])
    #print(punctation_dict[','])

    lexicon_dict = convert_to_relative_freq(lexicon_dict)
    punctation_dict = convert_to_relative_freq(punctation_dict)

    #print(lexicon_dict['der'])
    #print(punctation_dict[','])

    write_lexicon(lexicon_dict, 'dta_lexicon.txt')
    write_lexicon(punctation_dict, 'dta_punctuation.txt')


if __name__ == '__main__':
    main()



