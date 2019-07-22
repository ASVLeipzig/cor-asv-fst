import logging
import math
from os import listdir
import os.path
import pynini


def escape_for_pynini(s):
    '''
    Escapes a string for the usage in pynini. The following characters
    are prepended with a backslash:
    - the opening and closing square bracket,
    - the backslash itself.
    '''
    return s.replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')


def get_filenames(directory, suffix):
    '''
    Return all filenames following the scheme <file_id>.<suffix> in the
    given directory.
    '''

    return (f for f in listdir(directory) if f.endswith('.' + suffix))


def generate_content(directory, filenames):
    '''
    Generate tuples of file basename and file content string for given
    filenames and directory.
    '''

    for filename in filenames:
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield (filename.split('.')[0], line)


def load_pairs_from_file(filename):
    '''
    Load pairs of (line_ID, line) from a file.
    '''
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
    '''
    Load pairs of (line_ID, line) from a file. Each text file ending
    with `suffix` contains a line of text and the line ID is the file
    name without the suffix.
    '''
    filenames = get_filenames(directory, suffix)
    return list(generate_content(directory, filenames))


def load_lines_from_file(filename):
    '''
    Load text lines from file.
    '''
    lines = None
    with open(filename) as fp:
        lines = [line.rstrip() for line in fp]
    return lines


def load_wordlist_from_file(filename):
    '''
    Load wordlist from a CSV file (word <tab> frequency).
    '''
    result = {}
    with open(filename) as fp:
        for line in fp:
            try:
                word, freq = line.rstrip().split('\t')[:2]
                result[word] = int(freq)
            # ignore lines in wrong format
            # (less than two columns, second column is not a number etc.)
            except Exception:
                pass
    return result


def save_pairs_to_file(pairs, filename):
    '''
    Save pairs of (line_ID, line) to a file.
    '''
    with open(filename, 'w+') as fp:
        for p in pairs:
            fp.write('\t'.join(p) + '\n')


def save_pairs_to_dir(pairs, directory, suffix):
    '''
    Save pairs of (line_ID, line) to a directory.

    See the docstring of `load_pairs_from_dir` for an explanation of the
    format.
    '''
    for basename, string in pairs:
        filename = basename + '.' + suffix
        with open(os.path.join(directory, filename), 'w+') as fp:
            fp.write(string)


def convert_to_log_relative_freq(lexicon_dict, freq_threshold=2e-6):
    '''
    Convert counts of dict: word -> count into dict with relative
    frequencies: word -> relative_frequency.

    The entries with relative frequency lower than `freq_threshold` are
    dropped.
    '''

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


def transducer_from_dict(dictionary, unweighted=False):
    '''
    Given a dictionary of strings and weights, build a transducer
    accepting those strings with given weights.
    '''
    return pynini.string_map(\
        (escape_for_pynini(k),
         escape_for_pynini(k),
         str(w) if not unweighted else '0.0') \
        for k, w in dictionary.items())

