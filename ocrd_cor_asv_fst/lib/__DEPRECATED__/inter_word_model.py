import helper

import spacy

from functools import reduce


# TODO: improve this to become a comprehensive inter-word ngram model
# which also models sequences which did not appear in the traning data
# and has smoothing etc


def main():
    """This is an alternative for the bracket model of punctuation created in process_dta_data.py.
    Read GT. Count strings between words and split them to left/right part
    (relative to first space character) and also count complete strings
    ("middle" part).
    Needs to be extended to an ngram punctuation model."""

    # read gt data
    #path = '../dta19-reduced/traindata/'
    path = '../dta19-reduced/testdata/'
    gt_dict = helper.create_dict(path, 'gt')

    nlp = spacy.load('de')

    lines = list(gt_dict.values())

    inter_word_dict = {}

    #for line in lines[:10]:
    for line in lines:

        line = ' ' + line + ' '

        doc = nlp(line)

        tokens = list(doc)
        tokens_pos = [token.pos_ for token in doc]

        token_counter = 0
        token_position = 0

        inter_word_items = []

        item = ''

        # iterate over line and tokenized line, add punctuation and spaces
        # concatenated to the list of inter-word items
        for char in line:

            if token_counter == 0 and token_position == 0 and char == ' ':
                item += char

            if not token_counter == len(tokens):

                if char == tokens[token_counter].text[token_position]:

                    if tokens_pos[token_counter] == 'PUNCT':
                        #print("punct")
                        item += (char)

                    #print('same')

                    if token_position == len(tokens[token_counter].text) - 1:

                        if item != '' and tokens_pos[token_counter] != 'PUNCT' and char != ' ':
                            inter_word_items.append(item)
                            item = ''

                        token_counter += 1
                        token_position = 0

                    else:
                        token_position += 1

                else:
                    #print("space")
                    item += char

            else:

                if item != '':
                    item += char
                    inter_word_items.append(item)
                    #print(inter_word_items)
                    item = ''

        #print(inter_word_items)
        for item in inter_word_items:
            inter_word_dict[item] = inter_word_dict.setdefault(item, 0) + 1

    # distribute punctuation marks into left/middle/right
    left_part = {} # left of space
    middle_part = {} # complete string
    right_part = {} # space and right of space

    for item in inter_word_dict.items():

        # filter words containing alphanumerical characters
        if reduce(lambda x,y: x or y, list(map(lambda x: x.isalpha() or x.isnumeric(), item[0])), False):
            continue

        # split each string in left and right part and also save complete string
        middle_part[item[0]] = item[1]
        left_part[item[0].split(' ')[0]] = left_part.setdefault(item[0].split(' ')[0], 0) + item[1]
        right_part[item[0][item[0].find(' '):]] = right_part.setdefault(item[0][item[0].find(' '):], 0) + item[1]

    for dic in [middle_part, left_part, right_part]:
        if dic.get(' '):
            dic.pop(' ')
        if dic.get(''):
            dic.pop('')

    # print results
    #print("MIDDLE PART")
    #for item in middle_part.items():
    #    print(item)
    #print("LEFT PART")
    #for item in left_part.items():
    #    print(item)
    #print("RIGHT PART")
    #for item in right_part.items():
    #    print(item)

    # convert to relative frequency
    left_part   = helper.convert_to_relative_freq(left_part)
    middle_part = helper.convert_to_relative_freq(middle_part)
    right_part  = helper.convert_to_relative_freq(right_part)

    # save as list
    helper.write_lexicon(left_part, 'left_punctuation.txt')
    helper.write_lexicon(middle_part, 'middle_punctuation.txt')
    helper.write_lexicon(right_part, 'right_punctuation.txt')

    # construct transducers
    helper.save_transducer_from_txt('left_punctuation.txt', 'left_punctuation.hfst')
    helper.save_transducer_from_txt('middle_punctuation.txt', 'middle_punctuation.hfst')
    helper.save_transducer_from_txt('right_punctuation.txt', 'right_punctuation.hfst')

if __name__ == '__main__':
    main()

