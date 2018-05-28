from alignment.sequence import Sequence
import alignment
alignment.sequence.GAP_ELEMENT = "ε"
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner

import nltk
from nltk import ngrams


def file_to_dict(file_name):

    file_dict = {}

    with open(file_name) as f:
        for line in f.readlines():
            splitted = line.strip().split('\t')
            splitted = [x.strip() for x in splitted]

            if len(splitted) >= 5:
                file_dict['_'.join(splitted[0:4])] = splitted[4]

    return file_dict


def get_confusion_dicts():

    gt_file = ('gt.txt')
    raw_file = ('raw.txt')

    gt_dict = file_to_dict(gt_file)
    raw_dict = file_to_dict(raw_file)


    #gt_list = list(gt_dict.items())
    #
    #print(gt_list[0])
    #print(gt_dict['1035_0002.json_2_1_1'])


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

    for (gt_line, raw_line) in difference_list:
    #for (gt_line, raw_line) in difference_list[1:100]:

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
            #print(alignment)
            #print('Alignment score:', alignment.score)
            #print('Percent identity:', alignment.percentIdentity())

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



