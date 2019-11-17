from collections import namedtuple
import logging
import networkx as nx
from operator import itemgetter
import pynini
import time

from ocrd_models.ocrd_page import TextEquivType

from .helper import escape_for_pynini


def _print_paths(paths):
    paths_lst = [(output_str, float(weight)) \
                 for input_str, output_str, weight in paths.items()]
    paths_lst.sort(key=itemgetter(1))
    if not paths_lst:
        logging.debug('no paths!')
    else:
        for p in paths_lst:
            logging.debug('{}\t{}'.format(*p))


def split_input_string(string):
    '''Split an input string around whitespaces.'''
    return string.split()


def create_window(tokens):
    '''
    Create a window for the given input tokens (supplied as a list of
    strings).
    '''
    result = pynini.acceptor(escape_for_pynini(' '.join(tokens)))
    return result


def process_window(input_str, window_fst, model,
                   pruning_weight=5, rejection_weight=1.5):
    '''
    Compose a window input automaton with the model.
    '''
    t1 = time.time()
    window_fst.relabel_tables(
        new_isymbols=model[0].output_symbols(),
        new_osymbols=model[0].output_symbols())
    for fst in model:
        window_fst = pynini.compose(window_fst, fst)
        window_fst.project(project_output=True)
        window_fst.prune(weight=pruning_weight)
        window_fst.optimize()
    t3 = time.time()
    logging.debug('- composition: {}s'.format(t3-t1))
    # allow also identity for windows of length 1
    # (with weight `rejection_weight`)
    if ' ' not in input_str:
        # The formula:
        #    rejection_weight*(len(input_str)+2)
        # means that rejection_weight*2 is the initial cost of having an OOV
        # word (which is than more expensive with increasing length).
        # While discovered by accident, this turned out to work well as
        # a very naive OOV word model.
        window_fst.union(
            pynini.acceptor(
                escape_for_pynini(input_str),
                weight=rejection_weight*(len(input_str)+2)))
    t2 = time.time()
    logging.debug('Total processing time: {}s'.format(t2-t1))
    return window_fst


def recombine_windows(window_fsts):
    '''
    Recombine processed window FSTs (containing hypotheses for a given
    window) to a lattice, which is also represented as an FST.
    '''

    def _label(pos, length):
        return 'WIN-{}-{}'.format(pos, length)
    
    t1 = time.time()
    space_tr = pynini.acceptor(' ')

    # determine the input string length and max. window size
    # (TODO without iterating!!!)
    num_tokens = max(i for (i, j) in window_fsts)+1
    max_window_size = max(j for (i, j) in window_fsts)

    root = pynini.Fst()
    for i in range(num_tokens+1):
        s = root.add_state()
    root.set_start(0)
    root.set_final(num_tokens, 0)

    # FIXME refactor the merging of symbol tables into a separate function
    symbol_table = pynini.SymbolTable()
    for window_fst in window_fsts.values():
        symbol_table = pynini.merge_symbol_table(symbol_table,  window_fst.input_symbols())
        symbol_table = pynini.merge_symbol_table(symbol_table,  window_fst.output_symbols())
    for (pos, length), window_fst in window_fsts.items():
        label = _label(pos, length)
        sym = symbol_table.add_symbol(label)

    root.set_input_symbols(symbol_table)
    root.set_output_symbols(symbol_table)

    replacements = []
    for (pos, length), window_fst in window_fsts.items():
        label = _label(pos, length)
        sym = root.output_symbols().find(label)
        if pos+length < num_tokens:
            # append a space if this is not the last token, so that the final
            # string consists of tokens separated by spaces
            window_fst.concat(space_tr)
        replacements.append((label, window_fst))
        root.add_arc(pos, pynini.Arc(0, sym, 0, pos+length))

    result = pynini.replace(root, replacements)
    result.optimize()

    t2 = time.time()
    logging.debug('Recombining time: {}s'.format(t2-t1))

    return result


def lexicon_to_window_fst(lexicon_fst, words_per_window=2):
    '''
    Concatenate the lexicon FST `words_per_window` times, inserting
    spaces in between. The resulting FST accepts up to
    `words_per_window` words from the lexicon.
    '''
    result = lexicon_fst.copy()
    if words_per_window == 1:
        return result
    result.concat(pynini.acceptor(' '))
    result.closure(0, words_per_window-1)
    result.concat(lexicon_fst)
    return result


def lattice_shortest_path(lattice_fst):
    '''
    Extract the shortest path (i.e. with the lowest weight) from a
    lattice of hypotheses represented as an FST.
    '''
    return pynini.shortestpath(lattice_fst).stringify()


def combine_windows_to_graph(windows):
    '''
    Combine windows FSTs containing hypotheses for given windows to a
    graph of hypotheses in `nx.DiGraph` format, with decoding
    alternatives represented as `TextEquivType` at the edges. This is
    suitable for decoding data supplied in PageXML input format.

    The windows are passed as a dictionary:
        (starting_position, length) -> window_fst
    '''
    graph = nx.DiGraph()
    line_end_node = max(i+j for i, j in windows)
    graph.add_nodes_from(range(line_end_node + 1))
    for (i, j), fst in windows.items():
        start_node = i
        end_node = i + j
        paths = [(output_str, float(weight)) \
                 for input_str, output_str, weight in \
                     fst.paths().items()]
        if paths:
            for path in paths:
                logging.debug('({}, {}, \'{}\', {})'.format(\
                        start_node, end_node, path[0], pow(2, -path[1])))
            graph.add_edge(
                start_node, end_node, element=None,
                alternatives=[
                    TextEquivType(Unicode=path[0], conf=pow(2, -path[1])) \
                    for path in paths \
                ])
        else:
            logging.warning('No path from {} to {}.'.format(i, i+j))
    return graph


class FSTLatticeGenerator:
    '''
    This is the class responsible for generating lattices from input
    strings using the FST error model and lexicon. The lattices may be
    returned in two different output formats:
    - FST -- This allows for very fast search of a best path. It is the
             preferred format if no rescoring (with a language model) is
             applied afterwards.
    - networkx -- This returns the lattice as a `networkx.DiGraph`. It
                  is slower, but allows for rescoring.
    The output format has to be passed to the constructor, because
    working with two formats simultaneously is never necessary.
    '''

    def __init__(self, lexicon_file, error_model_file = None,
                 lattice_format = 'fst', **kwargs):
        # load all transducers and build a model out of them
        self.lexicon_fst = pynini.Fst.read(lexicon_file)
        self.window_fst = lexicon_to_window_fst(
            self.lexicon_fst,
            kwargs['words_per_window'])
        self.window_fst.arcsort()
        self.error_fst = pynini.Fst.read(error_model_file) \
                         if error_model_file \
                         else None
        self.rejection_weight = kwargs['rejection_weight']
        self.pruning_weight = kwargs['pruning_weight']
        self.max_window_size = 2                 # TODO expose as a parameter
        self.lattice_format = lattice_format

    def lattice_from_string(self, string):
        windows = {}
        tokens = split_input_string(string)
        for i in range(len(tokens)):
            for j in range(1, min(self.max_window_size+1, len(tokens)-i+1)):
                windows[(i,j)] = create_window(tokens[i:i+j])
        # compose each window with the model
        for (i, j) in windows:
            logging.debug('Processing window ({}, {})'.format(i, j))
            windows[(i,j)] = process_window(
                ' '.join(tokens[i:i+j]),
                windows[(i,j)],
                (self.error_fst, self.window_fst),
                pruning_weight=self.pruning_weight,
                rejection_weight=self.rejection_weight)
            _print_paths(windows[(i,j)].paths())

        # recombine the windows to a lattice represented in the desired format
        if self.lattice_format == 'fst':
            return recombine_windows(windows)
        elif self.lattice_format == 'networkx':
            return combine_windows_to_graph(windows)
        else:
            raise RuntimeError('Invaild lattice format: {}'\
                               .format(self.lattice_format))

