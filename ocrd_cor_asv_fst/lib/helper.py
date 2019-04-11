import hfst
import logging
import math
from os import listdir
import os.path
import pynini


def escape_for_pynini(s):
    '''
    Escapes a string for the usage in pynini. The following characters are
    prepended with a backslash:
    - the opening and closing square bracket,
    - the backslash itself.
    '''
    return s.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')


def save_transducer(filename, transducer, **kwargs):
    """Save hfst transducer to filename."""

    logging.debug('Saving transducer ({} states, {} arcs) to: {}'\
                  .format(transducer.number_of_states(),
                          transducer.number_of_arcs(),
                          filename))
    ostr = hfst.HfstOutputStream(filename=filename, **kwargs)
    ostr.write(transducer)
    ostr.flush()
    ostr.close()


def load_transducer(filename):
    """Load hfst transducer from filename."""

    logging.debug('Loading transducer from: {}'.format(filename))
    transducer = None
    istr = hfst.HfstInputStream(filename)
    while not istr.is_eof():
        transducer = istr.read()
    istr.close()

    return transducer


def get_filenames(directory, suffix):
    """Return all filenames following the scheme <file_id>.<suffix> in the given directory."""

    return (f for f in listdir(directory) if f.endswith('.' + suffix))


def generate_content(directory, filenames):
    """Generate tuples of file basename and file content string for given filenames and directory."""

    for filename in filenames:
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield (filename.split('.')[0], line)


# FIXME deprecated -- replace with load_pairs_from_dir()
def create_dict(directory, suffix):
    """
    Read files at the given directory path following the filename scheme 
    <file_id>.<suffix> and write them into a dict: file_id -> string content
    (each file is supposed to consist of one line of text).
    """

    result_dict = {}
    filenames = get_filenames(directory, suffix)
    content = generate_content(directory, filenames)
    for file_id, line in content:
        result_dict[file_id] = line
    return result_dict


def load_pairs_from_file(filename):
    results = []
    with open(filename) as fp:
        for i, line in enumerate(fp, 1):
            line_spl = line.rstrip().split('\t')
            if len(line_spl) >= 2:
                results.append(tuple(line_spl[:2]))
            else:
                logging.warning(\
                    '{}:{} -- line is not in two-column format: {}'\
                    .format(filename, i, line.rstrip()))
    return results


def load_pairs_from_dir(directory, suffix):
    filenames = get_filenames(directory, suffix)
    return list(generate_content(directory, filenames))


def save_pairs_to_file(pairs, filename):
    with open(filename, 'w+') as fp:
        for p in pairs:
            fp.write('\t'.join(p) + '\n')


def save_pairs_to_dir(pairs, directory, suffix):
    for basename, string in pairs:
        filename = basename + '.' + suffix
        with open(os.path.join(directory, filename), 'w+') as fp:
            fp.write(string)


def convert_to_log_relative_freq(lexicon_dict, freq_threshold=2e-6): # /13 # 6e-6/12 # 1e-5/11.5
    """Convert counts of dict: word -> count into dict with relative
    frequencies: word -> relative_frequency."""

    total_freq = sum(lexicon_dict.values())
    print('converting dictionary of %d tokens / %d types' % (total_freq, len(lexicon_dict)))
    for key in list(lexicon_dict.keys()):
        abs_freq = lexicon_dict[key]
        rel_freq = abs_freq / total_freq
        if abs_freq <= 3 and rel_freq < freq_threshold:
            print('pruning rare word form "%s" (%d/%f)' % (key, abs_freq, rel_freq))
            del lexicon_dict[key]
        else:
            lexicon_dict[key] = -math.log(rel_freq)
    return lexicon_dict


def convert_to_relative_freq(lexicon_dict, freq_threshold=2e-6): # /13 # 6e-6/12 # 1e-5/11.5
    """Convert counts of dict: word -> count into dict with relative
    frequencies: word -> relative_frequency."""
    # FIXME this function is deprecated -- remove as soon as no module uses it!

    total_freq = sum(lexicon_dict.values())
    print('converting dictionary of %d tokens / %d types' % (total_freq, len(lexicon_dict)))
    for key in list(lexicon_dict.keys()):
        abs_freq = lexicon_dict[key]
        rel_freq = abs_freq / total_freq
        if abs_freq <= 3 and rel_freq < freq_threshold:
            print('pruning rare word form "%s" (%d/%f)' % (key, abs_freq, rel_freq))
            del lexicon_dict[key]
        else:
            lexicon_dict[key] = rel_freq

    return lexicon_dict


def write_lexicon(lexicon_dict, filename, log=True):
    """Takes a lexicon_dict: word -> relative_frequency and writes it into
    a txt file at filename. If log is set, the negative logarithm of the
    relative_frequency is calculated (useful for transducer creation)."""

    lexicon_list = lexicon_dict.items()

    print('writing lexicon %s' % filename)
    with open(filename, 'w') as f:
        for word, freq in lexicon_list:
            if word == '':
                word = hfst.EPSILON
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
            if word == hfst.EPSILON:
                word = ''
            freq_list.append((word, word, float(frequency_class)))

    return freq_list


def transducer_from_list(freq_list):
    """Given a freq_list, construct a hfst transducer by disjuncting the
    entries of the list."""

    lexicon_transducer = hfst.HfstBasicTransducer()
    tok = hfst.HfstTokenizer()

    for i, (istr, ostr, weight) in enumerate(freq_list):
        lexicon_transducer.disjunct(tok.tokenize(istr), weight)
        #acceptor = hfst.regex(istr.replace('.', '%.') + ':' + ostr.replace('.', '%.') + '::' + str(weight))
        #lexicon_transducer.disjunct(acceptor)

        if i % 10000 == 0:
            print('wrote lemma nr %d' % i)

    print('wrote lemma nr %d' % i)
    return lexicon_transducer


def transducer_from_dict(dictionary):
    '''
    Given a dictionary of strings and weights, build a transducer accepting
    those strings with given weights.
    '''
    return pynini.string_map(\
        (escape_for_pynini(k), escape_for_pynini(k), str(w)) \
        for k, w in dictionary.items())


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


