import hfst
from os import listdir
import math


def save_transducer(filename, transducer):
    """Save hfst transducer to filename."""

    ostr = hfst.HfstOutputStream(filename=filename)
    ostr.write(transducer)
    ostr.flush()
    ostr.close()


def load_transducer(filename):
    """Load hfst transducer from filename."""

    transducer = None
    istr = hfst.HfstInputStream(filename)
    while not istr.is_eof():
        transducer = istr.read()
    istr.close()

    return transducer


def get_txt_files(directory, model):
    """Gives all files of form <file_id>.<model>.txt in given directory."""

    return (f for f in listdir(directory) if f.endswith('.' + model + '.txt'))


def get_content(directory, file_generator):
    """Yields content in files according to given file_generator in
    directory."""

    for x in file_generator:
        with open(directory + x) as f:
            yield [x.split('.')[0], f.read().strip()]


def create_dict(path, model):
    """Reads files at the given path with schema <file_id>.<model>.txt and writes
    them into a dict: file_id -> line (each file is supposed to consist of one
    line of text)."""

    result_dict = {}
    files = get_txt_files(path, model)
    content = get_content(path, files)
    for file_id, line in content:
        result_dict[file_id] = line
    return result_dict


def convert_to_relative_freq(lexicon_dict):
    """Convert counts of dict: word -> count into dict with relative
    frequencies: word -> relative_frequency."""

    summed_freq = sum(list(lexicon_dict.values()))
    for key in list(lexicon_dict.keys()):
        lexicon_dict[key] = lexicon_dict[key] / summed_freq

    return lexicon_dict


def write_lexicon(lexicon_dict, filename, log=True):
    """Takes a lexicon_dict: word -> relative_frequency and writes it into
    a txt file at filename. If log is set, the negative logarithm of the
    relative_frequency is calculated (useful for transducer creation)."""

    lexicon_list = list(lexicon_dict.items())

    with open(filename, 'w') as f:
        for word, freq in lexicon_list:
            if word == '':
                word = '@_EPSILON_SYMBOL_@'
            if log:
                f.write(word + '\t' + str(- math.log(freq)) + '\n')
            else:
                f.write(word + '\t' + str(freq) + '\n')

    return


def read_lexicon(filename):
    """Reads a lexicon of tab-separated word-frequency class pairs."""

    freq_list = []

    with open(filename) as f:
        for line in f.readlines():
            #word, frequency_class  = line.strip().split('\t')
            word, frequency_class  = line.split('\t')
            if len(word) > 1:
                word = word.strip()
            if word == '@_EPSILON_SYMBOL_@':
                word = ''
            freq_list.append((word, word, float(frequency_class)))

    return freq_list


def transducer_from_list(freq_list):
    """Given a freq_list, construct a hfst transducer by disjuncting the
    entries of the list."""

    lexicon_transducer = hfst.HfstBasicTransducer()
    tok = hfst.HfstTokenizer()

    counter = 0

    for (istr, ostr, weight) in freq_list:
        lexicon_transducer.disjunct(tok.tokenize(istr), weight)
        #acceptor = hfst.regex(istr.replace('.', '%.') + ':' + ostr.replace('.', '%.') + '::' + str(weight))
        #lexicon_transducer.disjunct(acceptor)

        counter += 1
        if counter % 10000 == 0:
            print(counter)

    return lexicon_transducer


def save_transducer_from_txt(input_txt, output_hfst):
    """Reads a lexicon of tab-separated word-frequency class pairs from
    input_txt, constructs a transducer from that list, and writes a hfst transducer to
    output_hfst."""

    print('Read', input_txt, '...')
    freq_list = read_lexicon(input_txt)
    print("Construct Transducer...")
    transducer = transducer_from_list(freq_list)
    transducer = hfst.HfstTransducer(transducer)
    save_transducer(output_hfst, transducer)
    print('Written to', output_hfst)


