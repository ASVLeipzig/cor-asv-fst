from collections import namedtuple
import logging
import networkx as nx
import pynini

from ocrd_models.ocrd_page import TextEquivType

from ..lib.sliding_window import \
    create_window, lexicon_to_window_fst, process_window, split_input_string, \
    _print_paths, recombine_windows


def combine_windows_to_graph(windows):
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
    # TODO docstrings

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
        self.beam_width = kwargs['beam_width']
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
                beam_width=self.beam_width,
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

