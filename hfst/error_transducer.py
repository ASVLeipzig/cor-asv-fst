import hfst
import math
import functools

import error_model2 as error_model


def preprocess_confusion_dict(confusion_dict):
    #"""Convert list of form ((input_string, output_string),
    #count) into list of form (((input_string, output_string),
    #relative_frequency), excluding infrequent errors,
    #maybe smoothing (not implemented)."""

    frequency_list = []

    items = confusion_dict.items()

    # count number of all occurrences
    s = [list(item[1].items()) for item in items]
    s = functools.reduce(lambda x, y: x + y, s, [])
    s = sum([x[1] for x in s])
    #print(s)

    # count number of ε substitutions
    epsilon_occ = sum([x[1] for x in confusion_dict.setdefault('ε', {}).items()])
    #epsilon_occ = sum([x[1] for x in confusion_dict['ε'].items()])
    #print(epsilon_occ)

    # set ε to ε transitions to number of all occurrences minus ε substitutions
    if epsilon_occ != 0:
        confusion_dict['ε']['ε'] = s - epsilon_occ

    for item in items:
        #print(item)

        input_str = item[0]
        substitutions = item[1].items()

        input_str_freq = sum([x[1] for x in substitutions])
        #print(input_str_freq)

        for sub in substitutions:
            frequency_list.append((input_str, sub[0], str(sub[1] / input_str_freq)))


    print(sorted(frequency_list, key=lambda x: x[2]))
    return frequency_list


def write_frequency_list(frequency_list, filename):

    with open(filename, 'w') as f:
        for entry in frequency_list:
            f.write('\t'.join(entry) + '\n')


def read_frequency_list(filename):

    freq_list = []

    with open(filename, 'r') as f:
        for line in f:
            instr, outstr, freq = line.strip('\n').split('\t')

            freq_list.append((instr, outstr, float(freq)))

    return freq_list


def transducer_from_list(confusion_list, frequency_class=False, identity_transitions=False):
    """Converts list of form (input_string, output_string, relative_frequency)
    into a weighted transducer performing the given transductions with the encountered
    probabilities. If frequency_class is True, instead of the relative
    frequency, a frequency class is given and no logarithm of the value
    will be performed for obtaining the weight."""

    if frequency_class:
        #confusion_list_log = list(map(lambda x: (x[0], x[1], x[2]), confusion_list))
        confusion_list_log = confusion_list
    else:
        confusion_list_log = list(map(lambda x: (x[0], x[1], -math.log(x[2])), confusion_list))

    fst_dict = {}

    for entry in confusion_list_log:
        instr, outstr, weight = entry

        # filter epsilon -> epsilon transition
        if (instr != 'ε' or outstr != 'ε'):
            #  filter identity transitions if identity_transitions=False
            if identity_transitions or instr != outstr:
                fst_dict[instr] = fst_dict.setdefault(instr, []) + [(outstr, weight)]

    confusion_fst = hfst.fst(fst_dict)
    confusion_fst.substitute('ε', hfst.EPSILON, input=bool, output=bool)

    return confusion_fst


def optimize_error_transducer(error_transducer):
    error_transducer.minimize()
    #error_transducer.repeat_star()
    error_transducer.remove_epsilons()
    error_transducer.push_weights_to_start()


def save_transducer(filename, transducer):
    ostr = hfst.HfstOutputStream(filename=filename)
    ostr.write(transducer)
    ostr.flush()
    ostr.close()


def load_transducer(filename):
    transducer = None
    istr = hfst.HfstInputStream(filename)
    while not istr.is_eof():
        transducer = istr.read()
    istr.close()

    return transducer


def main():

    n = 3

    confusion_dicts = error_model.get_confusion_dicts()
    confusion_list = preprocess_confusion_dict(confusion_dicts[n])
    write_frequency_list(confusion_list, 'confusion.txt')
    confusion_list = read_frequency_list('confusion.txt')
    error_transducer = transducer_from_list(confusion_list)
    optimize_error_transducer(error_transducer)

    #print(confusion_list)

    save_transducer('error_transducer_' + str(n) + '.hfst', error_transducer)
    error_transducer = load_transducer('error_transducer_' + str(n) + '.hfst')

    print(error_transducer)


if __name__ == '__main__':
    main()
