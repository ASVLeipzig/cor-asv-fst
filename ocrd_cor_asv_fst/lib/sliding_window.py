import hfst
import logging
from operator import itemgetter
import pynini
import tempfile
import time

from .helper import escape_for_pynini, load_transducer, save_transducer


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


def _hfst_create_window(tokens):
    '''
    Create a window for the given input tokens (supplied as a list of
    strings).
    '''
    return hfst.fst(' '.join(tokens))


def process_window_with_hfst(input_str, window_fst, model, n=10, rejection_weight=10):
    '''
    Compose a window input automaton with the model using HFST
    composition.
    '''
    t1 = time.time()
    for fst in model:
        window_fst.compose(fst)
    window_fst.minimize()
    window_fst.n_best(n)
    window_fst.output_project()
    if ' ' not in input_str:
        # allow also identity for windows of length 1
        # (with weight `rejection_weight`)
        window_fst.disjunct(hfst.fst((input_str, rejection_weight)))
    window_fst.minimize()
    t2 = time.time()
    logging.debug('states: {}'.format(window_fst.number_of_states()))
    logging.debug('Processing time: {}s'.format(t2-t1))
    return window_fst


def process_window_with_pynini(input_str, window_fst, model, n=10, rejection_weight=1.5):
    '''
    Compose a window input automaton with the model using Pynini
    composition.
    '''
    t1 = time.time()
    window_fst.relabel_tables(
        new_isymbols=model[0].output_symbols(),
        new_osymbols=model[0].output_symbols())
    for fst in model:
        window_fst = pynini.compose(window_fst, fst)
        window_fst.project(project_output=True)
        window_fst.prune(weight=5)
        window_fst.optimize()
    t3 = time.time()
    logging.debug('- composition: {}s'.format(t3-t1))
    if ' ' not in input_str:
        # allow also identity for windows of length 1
        # (with weight `rejection_weight`)
        window_fst.union(
            pynini.acceptor(
                escape_for_pynini(input_str),
                weight=rejection_weight*(len(input_str)+2)))
    t2 = time.time()
    # logging.debug('states: {}'.format(window_fst.number_of_states()))
    logging.debug('Total processing time: {}s'.format(t2-t1))
    return window_fst


def process_window(input_str, window_fst, model, rejection_weight=10):
    '''Compose a window input automaton with the model.'''
    if isinstance(model, tuple):
        return process_window_with_pynini(input_str, window_fst, model, rejection_weight)
    else:
        raise RuntimeError('Unknown model type: {}'.format(type(model)))


def recombine_windows(window_fsts):

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


def _hfst_recombine_windows(window_fsts):
    '''
    Recombine the window transducers into the transducer correcting the whole
    input string.
    '''

    t1 = time.time()

    # convert window fsts to basic transducers
    window_basic_fsts = { (i, j) : hfst.HfstBasicTransducer(window) \
                          for (i, j), window in window_fsts.items() }

    # determine the input string length and max. window size
    # (TODO without iterating!!!)
    num_tokens = max(i for (i, j) in window_fsts)+1
    max_window_size = max(j for (i, j) in window_fsts)

    result = hfst.HfstBasicTransducer()
    # add states:
    # 0 - the initial state
    # 1 - the final state
    result.add_state()
    assert result.states() == (0, 1)
    result.set_final_weight(1, 0.0)

    # create a dictionary translating the states of the window transducers to
    # the states of the resulting transducer
    state_dict = {}
    for (i, j), window in window_basic_fsts.items():
        for k in window.states():
            state_dict[(i, j, k)] = result.add_state()

    # add transitions leading to the first window of each size
    # (assuming that state 0 is initial in every HfstBasicTransducer!)
    for j in range(1, max_window_size+1):
        result.add_transition(
            0,
            hfst.HfstBasicTransition(
                state_dict[(0, j, 0)],
                hfst.EPSILON,
                hfst.EPSILON,
                0.0))

    for (i, j), window in window_basic_fsts.items():
        is_final_window = (i+j == num_tokens)
        for k in window.states():
            # add transitions inside the window FST
            for tr in window.transitions(k):
                result.add_transition(
                    state_dict[(i, j, k)],
                    hfst.HfstBasicTransition(
                        state_dict[(i, j, tr.get_target_state())],
                        tr.get_input_symbol(),
                        tr.get_output_symbol(),
                        tr.get_weight()))
            if window.is_final_state(k):
                if is_final_window:      # add a transition to the final state
                    result.add_transition(
                        state_dict[(i, j, k)],
                        hfst.HfstBasicTransition(
                            1,
                            hfst.EPSILON,
                            hfst.EPSILON,
                            window.get_final_weight(k)))
                else:                   # add transitions to next windows
                    new_i = i+j
                    for new_j in range(1, max_window_size+1):
                        if new_i+new_j <= num_tokens:
                            result.add_transition(
                                state_dict[(i, j, k)],
                                hfst.HfstBasicTransition(
                                    state_dict[(new_i, new_j, 0)],
                                    ' ',
                                    ' ',
                                    window.get_final_weight(k)))

    result = hfst.HfstTransducer(result)
    result.minimize()

    t2 = time.time()
    logging.debug('Recombining time: {}s'.format(t2-t1))

    return result


def process_string(string, model, max_window_size=2, rejection_weight=10):
    # create windows from the input string
    windows = {}
    tokens = split_input_string(string)
    for i in range(len(tokens)):
        for j in range(1, min(max_window_size+1, len(tokens)-i+1)):
            windows[(i,j)] = create_window(tokens[i:i+j])
    # compose each window with the model
    for (i, j) in windows:
        logging.debug('Processing window ({}, {})'.format(i, j))
        windows[(i,j)] = process_window(
            ' '.join(tokens[i:i+j]),
            windows[(i,j)], model,
            rejection_weight=rejection_weight)
        _print_paths(windows[(i,j)].paths())
    # recombine the windows
    final_fst = recombine_windows(windows)
    return final_fst


def lexicon_to_window_fst(lexicon_fst, words_per_window=2):
    result = lexicon_fst.copy()
    if words_per_window == 1:
        return result
    result.concat(pynini.acceptor(' '))
    result.closure(0, words_per_window-1)
    result.concat(lexicon_fst)
    return result


def lattice_shortest_path(lattice_fst):
    return pynini.shortestpath(lattice_fst).stringify()


##############################################################333
# The Corrector class
##############################################################333

class Corrector:
    def __init__(self, error_fst_file, lexicon_fst_file):
        pass

    def process_window(self, win_idx, window):
        return (win_idx, window)

