import hfst

import time
import math
import string
import unicodedata # for canonical normal form, for combining characters
from functools import reduce
import os
import sys
import traceback
import tempfile
import argparse
import logging

from composition import pyComposition
import helper

REJECTION_WEIGHT = 1.5 # weight assigned to all transitions in input transducer when disjoining with result transducer as fallback (see set_transition_weights); trade-off between over- and under-correction

def create_input_transducer(input_list): #(input_str, flag_encoder):
    """Takes tokenized input and creates a transducer accepting that string."""

    #fst_dict = {input_str: [(input_str, -math.log(1))]}
    #input_fst = hfst.fst(fst_dict)
    #input_fst = hfst.tokenized_fst(flag_encoder.tok.tokenize(input_str), weight=0.)
    #input_fst = hfst.tokenized_fst(tuple(input_list), weight=0.)
    basic_fst = hfst.HfstBasicTransducer()
    basic_fst.disjunct(tuple(zip(input_list, input_list)), 0.)
    input_fst = hfst.HfstTransducer(basic_fst)

    return input_fst


def prepare_input(input_str, window_size, flag_encoder, as_transducer=False):
    """
    Take the (long) input_string and split it at space characters
    into window strings with window_size words each. 
    For each window, make a transducer accepting that string if as_transducer,
    otherwise make a list of token strings. In both cases, insert 
    special flag transitions at window boundaries according to flag_encoder,
    so windows can be recombined afterwards.
    """

    # remove surrounding whitespace, then normalize to canonical form
    # (i.e. use pre-composed characters where available)
    windows = unicodedata.normalize('NFC', input_str.strip()).split(' ')

    # combine neighbouring "words", when one of the "words" is merely
    # a single punctuation character
    # TODO: ensure tokenization is consistent with create_lexicon (spacy + our rules)

    # TODO: if input is already an acceptor (i.e. hypotheses graph), we must still
    #       be able to do sliding window
    
    new_windows = []
    last_word = ''
    for word in windows:
        if len(word) == 1 and not word.isalnum() and not word in '—':
            if last_word != '':
                last_word = last_word + ' ' + word
            else:
                last_word = word
        elif last_word != '':
            new_windows.append(last_word)
            last_word = word
        else:
            last_word = word
    new_windows.append(last_word)
    windows = new_windows

    # create input transducers / input strings
    inputs = []
    for i in range(0, max(1, len(windows) - window_size + 1)):
        window_str = windows[i:i + window_size]
        window = [flag_encoder.encode(i)]
        for j, word in enumerate(window_str):
            # window.extend(word) # character by character
            # this breaks with combining characters like U+0364,
            # which Python treats like separate characters, and
            # unicodedata's NFC (canonical normal form) cannot
            # normalize.
            # so instead, we must fit those explicitly:
            for k, c in enumerate(word):
                if unicodedata.combining(c):
                    if k > 0:
                        window[-1] = window[-1]+c
                    else:
                        pass # or issue warning that a combining character was placed after a space?
                else:
                    window.append(c)
            window.append(' ')
            window.append(flag_encoder.encode(i+j+1))
        if as_transducer:
            # convert to transducer now (slower):
            inputs.append(create_input_transducer(window))
            #window_fst = hfst.HfstBasicTransducer()
            #window_fst.disjunct(tuple(zip(window, window)), 0) # TODO handle input weights
            #inputs.append(hfst.HfstTransducer(window_fst))
        else:
            # convert to transducer later (with OpenFST)
            inputs.append(window)
    
    return inputs


def compose_and_search(input, error_transducer, lexicon_transducer, result_num, flag_encoder, composition=None):
    """
    Perform composition of the input and given transducers,
    and search for result_num best paths in the result.
    Input can be either a list of strings (already tokenized)
    or a transducer.
    If composition is given, then execute both operations
    in OpenFST (which is faster). Otherwise use HFST.
    Afterwards, disjoin the resulting transducer with
    input_transducer, giving REJECTION_WEIGHT to all of
    its transitions (as fallback in case of empty result,
    and to prevent over-correction).
    """

    logging.debug('INPUT STRING')
    if isinstance(input, list):
        logging.debug(''.join(input))
        #input_transducer = create_input_transducer(input)
    else: # isinstance(input, hfst.HfstTransducer)
        print_shortest_path(input)
        input_transducer = input
    
    if composition != None: # compose using OpenFST

        #string = input_list_to_str(input_str, False)
        #input_file_path = 'input/' + string + '.hfst'

        #out = hfst.HfstOutputStream(filename=input_file_path, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        #out.write(input_fst)
        #out.flush()
        #out.close()

        ##composition.compose(input_str[1].encode())
        ##composition.compose(input_list_to_str(input_str, True).encode())
        ##composition.compose_file(string.encode())
        #composition.compose((input_list_to_str(input_str, True)).encode())

        #result_fst = et.load_transducer('output/' + string + '.fst')
        ##result_fst = et.load_transducer('output/' + input_str[1] + '.fst')
        ##result_fst = et.load_transducer(input_str + '.fst')

        if isinstance(input, list):
            filename = composition.correct_string('\n'.join(input)) # StringCompiler for SYMBOL splits at newline (fst_field_separator)
        else: # isinstance(input, hfst.HfstTransducer)
            #filename = composition.correct_transducer_string(input_transducer...) # not implemented yet
            with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-input') as f:
                #helper.save_transducer(f.name, input_transducer)
                write_fst(f.name, input)
                filename = composition.correct_transducer_file(f.name)

        result_fst = helper.load_transducer(filename)
        os.unlink(filename)

        global REJECTION_WEIGHT
        if REJECTION_WEIGHT < 0: # for ROC evaluation: no backoff_result in pyComposition
            result_fst.determinize() # fingers crossed!
            result_fst.remove_epsilons()
            #result_fst.minimize()

            results_fst = []
            for weight in [0., 0.01, 0.1, 0.2, 0.3, 0.7, 1., 1.3, 1.5, 1.7, 2., 3., 5., 10., 20.]:
                REJECTION_WEIGHT = weight
                one_result_fst = result_fst.copy()
                # disjunct result with input_fst, but with high transition weights (acts as a rejection threshold):
                one_result_fst.disjunct(set_transition_weights(input_transducer))
                one_result_fst.remove_epsilons()
                results_fst.append(one_result_fst)
            REJECTION_WEIGHT = -1
            result_fst = results_fst
        else:
            result_fst = [result_fst]

    else: # compose using HFST
        
        if isinstance(input, list):
            input_transducer = create_input_transducer(input)
        result_fst = input_transducer.copy()

        #logging.info("input_str: %s ", input_str)
        #print("compose transducers")
        #print("Input States: ", result_fst.number_of_states())

        result_fst.compose(error_transducer)
        result_fst.compose(lexicon_transducer)

        #print("Result States: ", result_fst.number_of_states())

        #print("Error States: ", error_transducer.number_of_states())
        #print("Lexicon States: ", lexicon_transducer.number_of_states())

        #result_fst.n_best(result_num)
        #results = input_fst.extract_paths(max_cycles=0, max_number=5, output='dict')

        #results = input_fst.extract_paths(max_number=result_num)

        #print("get best paths")

        result_fst.n_best(result_num)

        #print("Result States: ", result_fst.number_of_states())

        result_fst.prune() # necessary for determinize, otherwise might not have the twin property!
        result_fst.determinize()
        result_fst.remove_epsilons()
        #result_fst.minimize()

        # disjunct result with input_fst, but with high transition weights (acts as a rejection threshold):
        result_fst.disjunct(set_transition_weights(input_transducer))
        result_fst.remove_epsilons()

        result_fst = [result_fst]

    return result_fst


def set_transition_weights(fst):
    """Sets each transition of the given fst to REJECTION_WEIGHT."""
    global REJECTION_WEIGHT

    basic_fst = hfst.HfstBasicTransducer(fst)
    for state in basic_fst.states():
        for transition in basic_fst.transitions(state):
            transition.set_weight(REJECTION_WEIGHT) # 10.0 # hyperparameter!
    return hfst.HfstTransducer(basic_fst)


def print_output_paths(basic_fst):
    """Print the shortest path and five random paths in the basic_fst
    alongside their weight."""

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        #log_paths(hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0))
        log_paths(hfst.HfstTransducer(basic_fst).extract_shortest_paths())
        logging.debug('RANDOM PATHS:')
        log_paths(hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0))
        logging.debug('')


def print_shortest_path(basic_fst):
    """Print the shortest path alongside its weight."""
    
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        #log_paths(hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0))
        log_paths(hfst.HfstTransducer(basic_fst).extract_shortest_paths())

def log_paths(paths):
    for instr, outputs in paths.items():
        logging.debug('%s:', instr.replace(hfst.EPSILON, '□'))
        for outstr, weight in outputs:
            logging.debug('%s\t%f', outstr.replace(hfst.EPSILON, '□'), weight)

def get_flag_states(transducer, starting_state, flag_list):
    """
    Determine which states occur before which flags, 
    and which states are final states.
    For better performance, check only states
    reachable from given starting_state.
    """
    
    flag_state_dict = {} # flag_string -> list of states (e.g. @N.A@ -> [0])
    predecessor_dict = {} # state -> list of predecessor states
    predecessor_dict[0] = []
    final_states = []

    # flag_length = len(flag_list[0])
    # flag_starting_symbol = flag_list[0][0]

    visited = []
    queue = [starting_state]
    #flag_candidates = []
    
    while queue:

        current_state = queue.pop()
        visited.append(current_state)

        # remember final states
        if transducer.is_final_state(current_state):
            final_states.append(current_state)
        
        for transition in transducer.transitions(current_state):
            
            target_state = transition.get_target_state()
            predecessor_dict[target_state] = predecessor_dict.setdefault(target_state, []) + [current_state]
            
            # # current_state is a possible beginning of a flag
            # if transition.get_input_symbol() == flag_starting_symbol:
            #     flag_candidates.append(current_state)
            current_symbol = transition.get_input_symbol()
            if current_symbol in flag_list and current_state not in flag_state_dict.setdefault(current_symbol, []):
                flag_state_dict[current_symbol] += [current_state]
            
            # add target states to queue if not visited yet (and not added by other transition already)
            if target_state not in visited + queue:
                queue.append(target_state)
    
    #print('flag states:', flag_state_dict.items())
    #print('fst', transducer)

    return flag_state_dict, final_states, predecessor_dict


def merge_states(basic_transducer, state_list, predecessor_dict):
    '''
    Merge all states in the given state_list to a single state, 
    the first in the list. All incoming and outgoing transitions
    of the other states are redirected over this state.
    Forbid epsilon self-loops.
    '''

    def _incoming_transitions(state):
        '''Return a list of pairs `(predecessor, transition)` of incoming
           transitions for `state`.'''
        result = []
        for pred in predecessor_dict[state]:
            for t in basic_transducer.transitions(pred):
                if t.get_target_state() == state:
                    result.append((pred, t))
        return result

    def _try_adding_transition(state, transition):
        if transition.get_target_state() == state \
                and transition.get_input_symbol() != hfst.EPSILON \
                and transition.get_output_symbol() != hfst.EPSILON:
            logging.warn('merge would add loop at %d (%s:%s::%f)',
                         state, transition.get_input_symbol(),
                         transition.get_output_symbol(),
                         transition.get_weight())
        else:
            logging.debug('adding transition {} -> {}'\
                          .format(state, str(transition)))
            basic_transducer.add_transition(state, transition)

    def _remove_transition(state, transition):
        logging.debug('removing transition {} -> {}'\
                      .format(state, str(transition)))
        basic_transducer.remove_transition(state, transition)

    single_state = state_list[0]
    for state in state_list[1:]:
        if state == single_state:
            raise Exception('cannot merge state %d with itself' % state,
                            basic_transducer)
        # TODO: also remove dangling states (i.e. those from which the new
        # final states cannot be reached anymore)

        transitions_to_remove, transitions_to_add = [], []

        # incoming transitions
        for (p, t) in _incoming_transitions(state):
            transitions_to_remove.append((p, t))
            transitions_to_add.append((
                p,
                hfst.HfstBasicTransition(
                    single_state, t.get_input_symbol(),
                    t.get_output_symbol(), t.get_weight())))

        # outgoing transitions
        for t in basic_transducer.transitions(state):
            transitions_to_remove.append((state, t))
            transitions_to_add.append((
                single_state,
                hfst.HfstBasicTransition(
                    t.get_target_state(), t.get_input_symbol(),
                    t.get_output_symbol(), t.get_weight())))

        for (s, t) in transitions_to_remove:
            _remove_transition(s, t)
        for (s, t) in transitions_to_add:
            _try_adding_transition(s, t)
        predecessor_dict[state] = []

    # FIXME this is an unexpected side-effect
    # update flag_state_dict:
    for state in state_list[1:]:
        state_list.remove(state)

def write_fst(filename, fst):
    """Write fst to file."""

    #fst = hfst.HfstTransducer(fst)

    out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(fst)
    out.flush()
    out.close()

    return

def next_flag_state(flag_state_dict, start, flag_encoder):
    for i in range(start, flag_encoder.max_input_int):
        flag = flag_encoder.encode(i+1)
        if flag in flag_state_dict:
            return flag_state_dict[flag]
    raise Exception('cannot get next flag states for window merge\n' + traceback.format_exc())

def combine_results(result_list, window_size, flag_encoder):
    """Combine windows' results into one result transducer.
    
    Take a list of transducers, iteratively concatenate them 
    and merge states based on flags.
    """

    #for fst in result_list:
    #    print_output_paths(fst)

    flag_state_dict = {} # flag_string -> list of states (e.g. @N.A@ -> [0])
    final_states = []

    #write_fst('output/window_results.hfst', result_list)

    # start with first window
    starting_fst = result_list[0].copy()
    #starting_fst.output_project()
    #starting_fst.minimize()
    #starting_fst.remove_epsilons()

    #write_fst('output/starting_fst.hfst', starting_fst)
    result_fst = hfst.HfstBasicTransducer(starting_fst)
    starting_fst.output_project()
    
    logging.debug('WINDOW RESULT PATHS')
    print_shortest_path(starting_fst)
    
    flag_state_dict, final_states, predecessor_dict = get_flag_states(result_fst, 0, flag_encoder.flag_list)
    #print('flag states', list(flag_state_dict.items()))
    #print('final states', final_states)

    # merge states in initial transducer
    merge_list = next_flag_state(flag_state_dict, 0, flag_encoder)
    merge_states(result_fst, merge_list, predecessor_dict)

    #print('PARTIAL RESULT PATHS')
    #print_output_paths(starting_fst)
    for i, fst in enumerate(result_list[1:]):

        fst.output_project()
        #fst.minimize()
        #fst.remove_epsilons()

        partial_fst = hfst.HfstBasicTransducer(fst)
        
        logging.debug('WINDOW RESULT PATHS')
        print_shortest_path(partial_fst)

        #  remove final states in result fst
        for state in final_states:
            result_fst.remove_final_weight(state)

        # determine append states and set them final
        append_states = next_flag_state(flag_state_dict, i, flag_encoder)
        for state in append_states:
            result_fst.set_final_weight(state, 0.0)

        #print('append_states', append_states)

        #logging.debug('BEFORE CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)
        #write_fst('output/before_concatenation.hfst', hfst.HfstTransducer(result_fst))

        # concatenate result fst and partial result
        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)

        result_fst.concatenate(partial_fst)
        #write_fst('output/partial_result.hfst', hfst.HfstTransducer(result_fst))

        ##print("number of states :", result_fst.number_of_states())
        #print('PARTIAL RESULT PATHS')
        #print_output_paths(partial_fst)
        #logging.debug('AFTER CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)
        ##result_fst.n_best(100)

        result_fst = hfst.HfstBasicTransducer(result_fst)

        # update final_states and flag_state_dict
        flag_state_dict, final_states, predecessor_dict = get_flag_states(result_fst, 0, flag_encoder.flag_list)

        #print('before merge')
        #print('flag states', flag_state_dict.items())
        #print('final states', final_states)

        #write_fst('output/before_merge.hfst', hfst.HfstTransducer(result_fst))
        #print('before merge', result_fst)

        # TODO: besuche nur die neuen Zustände und update das existierende
        # dictionary (get_flag_states Startzustand übergeben); prüfe, ob
        # die Zustands-IDs durch die Konkatenation nicht geändert werden;
        # nur dann kann das funktionieren
        # TODO: keine complete_merge, sondern mit merge_states nur die
        # betroffenen neuen flag_states mergen
        complete_merge(result_fst, flag_encoder)
        flag_state_dict, final_states, predecessor_dict = get_flag_states(result_fst, 0, flag_encoder.flag_list)

        #print('after merge', result_fst)
        #write_fst('output/after_merge.hfst', hfst.HfstTransducer(result_fst))

        #print('after merge')
        #print('flag states', flag_state_dict.items())
        #print('final states', final_states)

        ## merge word borders
        ## merge_word_borders(result_fst, path_dict, predecessor_dict)

        #logging.debug('AFTER MERGE RESULT PATHS')
        #print_output_paths(result_fst)


        ##print('before merge', result_fst)
        #write_fst('output/before_merge.hfst', result_fst)

        ## merge corresponding states
        #merge_list = flag_state_dict[flag_encoder.encode(i+1)]
        #merge_states(result_fst, merge_list, predecessor_dict)

        ## update flag_state_dict
        #flag_state_dict[flag_encoder.encode(i+1)] = [merge_list[0]]

        ##print('after merge', result_fst)
        #write_fst('output/after_merge.hfst', result_fst)

    #sys.exit(0)

    return result_fst


def create_result_transducer(input_str, window_size, words_per_window, error_transducer, lexicon_transducer, result_num, flag_encoder, composition=None):
    """
    Prepare the input_str for a given window_size and 
    perform composition and search of result_num best paths
    on each of the windows.
    Combine the windows' results to a single transducer.
    """

    #lexicon_transducer.repeat_n(words_per_window)

    start = time.clock()

    # as_transducer=True: always use create_input_transducer from sliding_window.py
    # as_transducer=False: if composition!=None (main:openfst==True),
    #                      then use create_input_transducer from composition_cpp.cpp
    #                           (which does not work with REJECTION_WEIGHT < 0, though)
    #                      else as above
    input_transducers = prepare_input(input_str, window_size, flag_encoder, as_transducer=False)
    output_transducers = []
    for i, input_transducer in enumerate(input_transducers):
        result = compose_and_search(input_transducer, error_transducer, lexicon_transducer, result_num, flag_encoder, composition=composition)
        output_transducers.append(result)
    
    after_composition = time.clock()

    #complete_output = combine_results(output_list, window_size)
    #complete_output = remove_redundant_paths(complete_output)

    complete_outputs = []
    for i in range(len(output_transducers[0])):
        complete_output = combine_results(list(map(lambda x: x[i], output_transducers)), window_size, flag_encoder)
        complete_output = hfst.HfstTransducer(complete_output)
        complete_outputs.append(complete_output)
    
    after_combination = time.clock()

    logging.info('Composition Time: %f', after_composition - start)
    logging.info('Combination Time: %f', after_combination - after_composition)

    return complete_outputs


def get_edit_space_transducer(flag_encoder):
    """Reads a space and a flag diacritic for marking word borders and
    removes it. Needed to replace space to epsilon edits in the error
    transducer to handle merges of two words."""

    remove_diacritics_transducer = hfst.HfstBasicTransducer()
    for flag in flag_encoder.flag_list:
        remove_diacritics_transducer.disjunct(flag_encoder.tok.tokenize(flag, ''), 0.0)

    #print(remove_diacritics_transducer)
    remove_diacritics_transducer = hfst.HfstTransducer(remove_diacritics_transducer)
    remove_diacritics_transducer.optionalize()

    remove_space_transducer = hfst.regex('% :0') # space to epsilon
    remove_space_transducer.concatenate(remove_diacritics_transducer)
    remove_space_transducer.minimize()

    #print(remove_space_transducer)

    return remove_space_transducer


def get_flag_acceptor(flag_encoder):
    """Transducer that accepts the flags of the flag encoder. It is needed
    before and after each concatenated lexicon transducer to mark the word
    borders."""

    flag_acceptor = hfst.HfstBasicTransducer()
    for flag in flag_encoder.flag_list:
        flag_acceptor.disjunct(flag_encoder.tok.tokenize(flag), 0.0)
    
    #print('Flag Acceptor: ', flag_acceptor)
    flag_acceptor = hfst.HfstTransducer(flag_acceptor)
    #flag_acceptor.optionalize() # wrong, and extremely time costly!

    return flag_acceptor

    # Multichar
    # (for obtaining flags that are single symbols;
    # does not work when using OpenFST, since it removes these flags)

    flag_acceptor = hfst.HfstBasicTransducer()
    flag_acceptor.add_state(1)
    flag_acceptor.set_final_weight(1, 0.0)

    for symbol in flag_list:
        flag_acceptor.add_transition(0, 1, symbol, symbol, 0.0)

    flag_acceptor = hfst.HfstTransducer(flag_acceptor)
    flag_acceptor.optionalize()

    #print('Flag Acceptor: ', flag_acceptor)

    return flag_acceptor

def lexicon_add_compounds(lexicon_transducer, composition_depth):
    # FIXME: to get a better approximation of compounding rules,
    #        we really need word classes; in the very least, we
    #        should allow only noun compounds by guessing from capitalization
    #        (false positives are easier to avoid with only 1 class
    #         and nouns are easily detectable and most productive)
    # FIXME: the current lexicon contains each form both capitalized and downcased
    #        (so we get lots of false positives again); what we should do instead, 
    #        is keeping only the most frequent of both forms after lexicon creation,
    #        but dynamically allowing titlecase after sentence punctuation or at
    #        the start of a line in lexicon transducer
    # FIXME: similar problem for hyphenated and abbreviated words
    # 
    # Fugenmorpheme (Komposition) / most common German compound infixes
    # e.g. hilf-s-motor|gesetz-es-treu|schreib[en]-maschine|beuge[n]-haft:
    #
    #old solution (did not respect case):
    # connect_composition = hfst.regex('"-"|s|e|{es}|{en}:0|{en}:e')
    # connect_composition.optionalize()
    # connect_composition.concatenate(lexicon_transducer)
    # connect_composition.repeat_n(composition_depth - 1)
    # lexicon_transducer.concatenate(connect_composition)
    #
    #new solution (case-based noun detection, downcasing for infixes, but not for hyphenated form):
    connect_composition = hfst.regex('0|s|e|{es}')
    compound_base = lexicon_transducer.copy()
    downcase = hfst.regex('|'.join(up + ':' + lo for up, lo in zip(string.ascii_uppercase + "ÄÖÜ", string.ascii_lowercase + "äöü")))
    downcase.concatenate(hfst.regex('?+'))
    compound_base.compose(downcase)
    connect_composition.concatenate(compound_base)

    hyphen_composition = hfst.regex('"-"')
    compound_base = lexicon_transducer.copy()
    uppercase = hfst.regex('|'.join(list(string.ascii_uppercase + "ÄÖÜ")))
    uppercase.concatenate(hfst.regex('?+'))
    compound_base.compose(uppercase)
    hyphen_composition.concatenate(compound_base)

    connect_composition.disjunct(hyphen_composition)

    connect_composition.repeat_n_to_k(1, composition_depth - 1)

    compound = lexicon_transducer.copy()
    upcase = hfst.regex('|'.join(lo + ':' + up for up, lo in zip(string.ascii_uppercase + "ÄÖÜ", string.ascii_lowercase + "äöü")))
    upcase.concatenate(hfst.regex('?+')) # for nonnoun-noun
    upcase.disjunct(uppercase) # for noun-noun
    compound.compose(upcase)

    compound.concatenate(connect_composition)
    compound.minimize()

    lexicon_transducer.disjunct(compound)
    return lexicon_transducer


def load_transducers_bracket(error_file,
    punctuation_file,
    lexicon_file,
    open_bracket_file,
    close_bracket_file,
    flag_encoder,
    composition_depth=1,
    words_per_window=3,
    morphology_file=None):
    """
    Load transducers for the bracket model, including the error FST,
    lexicon FSA, morphology FST, punctuation FSA, and opening/closing-bracket FSA.
    
    Amend the error FST, also trying to delete flags when deleting spaces.
    
    Amend the lexicon FSA for compound words, and for decomposed umlauts.
    Compose with morphology to extend further, and project to an FSA.
    
    Concatenate the lexicon FSAs to a single-token FSA, then repeat for
    given words_per_window, synchronizing each word with flags.
    """
    
    # load transducers
    flag_acceptor = get_flag_acceptor(flag_encoder)
    space_transducer = hfst.regex('% :% ')
    error_transducer = helper.load_transducer(error_file)
    
    # when deleting spaces, try to also delete flags:
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))
    
    # ensure error transducer already contains flag symbols:
    alphabet = error_transducer.get_alphabet()
    for flag in flag_encoder.flag_list:
        if flag not in alphabet:
            logging.warning('error transducer did not have flag %s yet', flag)
            error_transducer.insert_to_alphabet(flag)
    
    punctuation_transducer = helper.load_transducer(punctuation_file)
    open_bracket_transducer = helper.load_transducer(open_bracket_file)
    close_bracket_transducer = helper.load_transducer(close_bracket_file)
    
    punctuation_transducer.optionalize()
    open_bracket_transducer.optionalize()
    close_bracket_transducer.optionalize()
    
    lexicon_transducer = helper.load_transducer(lexicon_file)
    
    # add compounds to lexicon:
    if composition_depth > 1:
        lexicon_transducer = lexicon_add_compounds(lexicon_transducer, composition_depth)
    
    # add derivation+inflection morphology to lexicon
    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)
    
    # allow both decomposed (as in lexicon file) and precomposed (modern) umlaut variants:
    precompose_transducer = hfst.regex('[aͤ:ä|oͤ:ö|uͤ:ü|Aͤ:Ä|Oͤ:Ö|Uͤ:Ü|?]*')
    lexicon_transducer.compose(precompose_transducer)

    # make sure above lexical transductions never enter the result:
    lexicon_transducer.output_project()

    # synchronize with left window boundary:
    result_lexicon_transducer = flag_acceptor.copy()
    
    # combine transducers to single-token lexicon transducer:
    result_lexicon_transducer.concatenate(open_bracket_transducer) # (optional)
    result_lexicon_transducer.concatenate(lexicon_transducer) # includes dash and numbers
    result_lexicon_transducer.concatenate(punctuation_transducer) # (optional)
    result_lexicon_transducer.concatenate(close_bracket_transducer) # (optional)
    result_lexicon_transducer.concatenate(space_transducer)
    
    # repeat single-token lexicon transducer according to maximum words per window:
    result_lexicon_transducer.repeat_n_to_k(1, words_per_window)
    
    # synchronize with right window boundary:
    result_lexicon_transducer.concatenate(flag_acceptor)

    return error_transducer, result_lexicon_transducer



def load_transducers_preserve_punctuation(error_file,
    punctuation_file,
    lexicon_file,
    flag_encoder,
    composition_depth=1,
    words_per_window=3,
    morphology_file=None):
    """
    Load transducers for the punctuation-preserving model, 
    including the error FST, lexicon FSA, morphology FST, and
    punctuation FSA (regardless of left/right context).
    
    Amend the error FST, also trying to delete flags when deleting spaces.
    (This error FST must never change any characters into punctuation
    characters.)
    
    Amend the lexicon FSA for compound words, and for decomposed umlauts.
    Compose with morphology to extend further, and project to an FSA.
    
    Concatenate the lexicon FSAs to a single-token FSA, then repeat for
    given words_per_window, synchronizing each word with flags.
    """
    
    # load transducers
    flag_acceptor = get_flag_acceptor(flag_encoder)
    space_transducer = hfst.regex('% :% ')
    error_transducer = helper.load_transducer(error_file)
    
    # when deleting spaces, try to also delete flags:
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))
    
    # ensure error transducer already contains flag symbols:
    alphabet = error_transducer.get_alphabet()
    for flag in flag_encoder.flag_list:
        if flag not in alphabet:
            logging.warning('error transducer did not have flag %s yet', flag)
            error_transducer.insert_to_alphabet(flag)
    
    punctuation_transducer = helper.load_transducer(punctuation_file)
    #punctuation_transducer.optionalize()
    #open_bracket_transducer = helper.load_transducer(open_bracket_file)
    #open_bracket_transducer.optionalize()
    #close_bracket_transducer = helper.load_transducer(close_bracket_file)
    #close_bracket_transducer.optionalize()
    
    lexicon_transducer = helper.load_transducer(lexicon_file)
    
    # add composed words to lexicon
    if composition_depth > 1:
        lexicon_transducer = lexicon_add_compounds(lexicon_transducer, composition_depth)
    
    # add morphology to lexicon
    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)
    
    # allow both decomposed (as in lexicon file) and precomposed (modern) umlaut variants:
    precompose_transducer = hfst.regex('[aͤ:ä|oͤ:ö|uͤ:ü|Aͤ:Ä|Oͤ:Ö|Uͤ:Ü|?]*')
    lexicon_transducer.compose(precompose_transducer)
    
    # make sure above lexical transductions never enter the result:
    lexicon_transducer.output_project()
    
    # synchronize with left window boundary:
    result_lexicon_transducer = flag_acceptor.copy()
    
    # combine transducers to single-token lexicon transducer:
    #result_lexicon_transducer.concatenate(open_bracket_transducer)
    result_lexicon_transducer.concatenate(punctuation_transducer) # both left and right contexts
    result_lexicon_transducer.concatenate(lexicon_transducer) # includes dash and numbers, has no edits removing punctuation
    result_lexicon_transducer.concatenate(punctuation_transducer) # both left and right contexts
    #result_lexicon_transducer.concatenate(close_bracket_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    
    # repeat single-token lexicon transducer according to maximum words per window:
    result_lexicon_transducer.repeat_n_to_k(1, words_per_window)
    
    # synchronize with right window boundary:
    result_lexicon_transducer.concatenate(flag_acceptor)

    return error_transducer, result_lexicon_transducer


def load_transducers_inter_word(error_file,
    lexicon_file,
    punctuation_left_file,
    punctuation_right_file,
    flag_encoder,
    words_per_window = 3,
    composition_depth = 1,
    morphology_file=None):
    """
    Load transducers for the inter-word model, including the error FST,
    lexicon FSA, morphology FST, and left/right-punctuation FSA.
    
    Amend the error FST, also trying to delete flags when deleting spaces.
    
    Amend the lexicon FSA for compound words, and for decomposed umlauts.
    Compose with morphology to extend further, and project to an FSA.
    
    Concatenate the lexicon FSAs to a single-token FSA, then repeat for
    given words_per_window, synchronizing each word with flags.
    """

    # TODO: handle flag diacritics; the construction of the lexicon is not
    # correct anymore; the punctuation_right_transducer as constructed
    # contains a space character at the beginning which shouldn't be there,
    # since the space is placed at the end of each window (with no
    # punctuation characters after that); since this model should be
    # changed to a complete punctuation ngram model anyway, the corrections
    # can be made on that occasion

    # load transducers
    flag_acceptor = get_flag_acceptor(flag_encoder)
    space_transducer = hfst.regex('% :% ')
    error_transducer = helper.load_transducer(error_file)
    
    # when deleting spaces, try to also delete flags:
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))
    
    # ensure error transducer already contains flag symbols:
    alphabet = error_transducer.get_alphabet()
    for flag in flag_encoder.flag_list:
        if flag not in alphabet:
            logging.warning('error transducer did not have flag %s yet', flag)
            error_transducer.insert_to_alphabet(flag)
    
    punctuation_left_transducer = helper.load_transducer(punctuation_left_file)
    punctuation_right_transducer = helper.load_transducer(punctuation_right_file)
    
    # compensate for outdated model (see TODO above):
    punctuation_right_transducer.compose(hfst.regex('% :0 ?*'))
    punctuation_right_transducer.output_project()

    punctuation_left_transducer.optionalize()
    punctuation_right_transducer.optionalize()

    lexicon_transducer = helper.load_transducer(lexicon_file)
    
    # add compounds to lexicon:
    if composition_depth > 1:
        lexicon_transducer = lexicon_add_compounds(lexicon_transducer, composition_depth)
    
    # add derivation+inflection morphology to lexicon
    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)
    
    # allow both decomposed (as in lexicon file) and precomposed (modern) umlaut variants:
    precompose_transducer = hfst.regex('[aͤ:ä|oͤ:ö|uͤ:ü|Aͤ:Ä|Oͤ:Ö|Uͤ:Ü|?]*')
    lexicon_transducer.compose(precompose_transducer)
    
    # make sure above lexical transductions never enter the result:
    lexicon_transducer.output_project()
    
    # synchronize with left window boundary:
    result_lexicon_transducer = flag_acceptor.copy()
    
    # combine transducers to single-token lexicon transducer:
    result_lexicon_transducer.concatenate(lexicon_transducer)
    result_lexicon_transducer.concatenate(punctuation_left_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    result_lexicon_transducer.concatenate(punctuation_right_transducer)
    
    # repeat single-token lexicon transducer according to maximum words per window:
    result_lexicon_transducer.repeat_n_minus(words_per_window-1)
    result_lexicon_transducer.concatenate(flag_acceptor)
    result_lexicon_transducer.concatenate(lexicon_transducer)
    result_lexicon_transducer.concatenate(punctuation_left_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    
    # synchronize with right window boundary:
    result_lexicon_transducer.concatenate(flag_acceptor)
    
    return error_transducer, result_lexicon_transducer


def complete_merge(basic_fst, flag_encoder):
    """
    Merge all states that are predecessors of the special merge flags.
    (This must be done to combine windows of size 1 and 2, or
     same-size adjacent windows.)
    """

    flag_state_dict, final_states, predecessor_dict = get_flag_states(basic_fst, 0, flag_encoder.flag_list)

    for flag, flag_states in flag_state_dict.items():
        #logging.debug('merging flag_states %s for flag %s', str(flag_states), flag)
        merge_states(basic_fst, flag_states, predecessor_dict)

    #logging.debug('merging final_states %s', str(final_states))
    merge_states(basic_fst, final_states, predecessor_dict)

    if len(final_states) > 1:
        for state in final_states[1:]:
            basic_fst.remove_final_weight(state)

    return


def window_size_1(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):
    """Apply correction with window size 1. Each window may contain up to
    three (merged) words."""

    window_1 = create_result_transducer(\
        input_str, 1, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    return window_1


def window_size_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):
    """Apply correction with window size 2. Each window may contain up to
    three (merged) words."""

    if ' ' not in input_str:
        return window_size_1(input_str, error_transducer, lexicon_transducer, result_num)

    window_2 = create_result_transducer(\
        input_str, 2, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    return window_2


def window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):
    """Apply correction with window sizes 1 and 2, merge both results to a
    single transducer."""

    if ' ' not in input_str:
        return window_size_1(\
            input_str, error_transducer, lexicon_transducer, flag_encoder, result_num, composition)

    # create results transducers for window sizes 1 and 2, disjunct and merge states

    windows_1 = create_result_transducer(
        input_str, 1, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    windows_2 = create_result_transducer(
        input_str, 2, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)

    complete_outputs = []
    for i, (window_1, window_2) in enumerate(zip(windows_1, windows_2)):
        window_1.disjunct(window_2)
        complete_output_basic = hfst.HfstBasicTransducer(window_1)
        
        before_merge = time.clock()
        
        complete_merge(complete_output_basic, flag_encoder)
        
        after_merge = time.clock()
        logging.info('Merge Time: %f', after_merge - before_merge)
        
        complete_output = hfst.HfstTransducer(complete_output_basic)
        #complete_output.minimize()

        complete_outputs.append(complete_output)
    
    return complete_outputs


def remove_flags(fst, flag_encoder):
    """
    Remove flags introduced for sliding window method from given fst, 
    given the corresponding flag_encoder. This is performed by replacing
    each of the flag transitions with an epsilon transition.
    """

    basic_fst = hfst.HfstBasicTransducer(fst)
    for flag in flag_encoder.flag_list:
        basic_fst.substitute(flag, hfst.EPSILON, input=True, output=True)
    
    #basic_fst.remove_symbols_from_alphabet(flag_encoder.flag_list)
    fst = hfst.HfstTransducer(basic_fst)
    fst.remove_epsilons()
    
    return fst


class FlagEncoder:
    """
    Endode and decode integers (between first_input_int and max_input_int)
    to special flag symbols used in the transducer. Flags serve as anchors
    for re-synchronization in the sliding window construction: 
    They are introduced when splitting the input string into windows, they
    are never edited and always accepted between words/windows, and they are
    subsequently removed after merging states of the resulting window transducers.
    """
    # TODO: allow a wider range of integers to encode, for example by using
    # two consecutive alphabetical characters (26*26 = 676)

    def __init__(self):

        alphabet = list(string.ascii_uppercase)

        self.int_char_dict = {}
        self.char_int_dict = {}

        for character in alphabet:
            self.int_char_dict[ord(character)] = character
            self.char_int_dict[character] = ord(character)

        self.first_int = ord(alphabet[0])
        self.max_int = ord(alphabet[-1])

        self.first_input_int = 0
        self.max_input_int = 25

        self.flag_list = []
        self.tok = hfst.HfstTokenizer()
        for i in range(self.first_input_int, self.max_input_int + 1):
            flag = self.encode(i)
            self.flag_list.append(flag)
            self.tok.add_multichar_symbol(flag)

    def encode(self, num):
        if num > self.max_int:
            raise RuntimeError('Flag encoding of number higher than 25 requested!')
            return ''
        character_num = self.first_int + num
        return '@N.' + chr(character_num) + '@'

    def decode(self, flag):
        flag = flag[1:-1]
        splitted = flag.split('.')
        return ord(splitted[1]) - self.first_int


def main():

    start = time.clock()

    # command line options
    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst one-shot tool')
    parser.add_argument('inputline', metavar='STRING', default=u"Philoſophenvon Fndicn dur<h Gricche nland bis", help='specify input string')
    parser.add_argument('-P', '--punctuation', metavar='MODEL', type=str, choices=['bracket', 'lm', 'preserve'], default='bracket', help='how to model punctuation between words (bracketing rules, inter-word language model, or keep unchanged)')
    parser.add_argument('-W', '--words-per-window', metavar='WORDS', type=int, default=3, help='maximum number of words in one window')
    parser.add_argument('-R', '--result-num', metavar='RESULTS', type=int, default=10, help='result paths per window')
    parser.add_argument('-D', '--composition-depth', metavar='DEPTH', type=int, default=1, help='number of lexicon words that can be concatenated')
    parser.add_argument('-J', '--rejection-weight', metavar='WEIGHT', type=float, default=1.5, help='transition weight for unchanged input window')
    parser.add_argument('-L', '--log-level', metavar='LEVEL', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG', help='verbosity of logging output (standard log levels)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.getLevelName(args.log_level))
    
    flag_encoder = FlagEncoder()

    #input_str = "bIeibt"
    #input_str = "{CAP}unterlagen"
    #input_str = "fur"
    #input_str = "das miüssenwirklich seHr sclnöne bla ue Schuhe seln"
    #input_str = "Das miüssenwirklich seHr sclnöne bla ue sein"
    #input_str = "miü ssen miü ssen wirklich"# seHr sclnöne bla ue Schuhe seln"
    #input_str = "sclnöne bla ue sein"
    #input_str = "Philoſophen von Fndicn dur<h Gricchenland bis"
    #input_str = "wirrt ſah fie um ſh, und als fie den Mann mit dem"
    #input_str = 'Man frage weiter das ganze liebe Deutſchland'
    #input_str = 'in Dir den Pater Medardus wieder zu erken—'
    #input_str = '3053'
    #input_str = "trachtet; ih ſehe ſeine weite ausgetáſeléte"
    #input_str = "trachtet; ih ſehe ſeine weite ausgetáſeléte—"
    #input_str = 'heit Anderer?« fragte ich lächelnd. »Mehr als jedes'
    #input_str = '„ Zawort“! feuchet ſie im Grimme.'
    # korrekt: „Jawort“! keuchet ſie im Grimme.
    #input_str = 'Sehstes Kapitel.'
    #input_str.strip('\n\u000C')
    #input_str = 'er das Fräulein auf einem ſchönen Zelter unten rider'
    #input_str = 'er das Fräulein auf einem ſchönen Zelter unten rider'
    #input_str = "Philoſophenvon Fndicn dur<h Gricche nland bis"

    # TODO: This test sentence has problems removing the flag strings when
    # using the any_punctuation_no_space model.
    # Further investigate the causes of the problem and write test(s) for
    # this.
    # Using any_punctuation_with_space model, an additional space character
    # is introduced, leading to two consecutive space characters.
    # Has the flag string been edited?
    #input_str = "\ opf. Mir wurde plöblich fo klar, — jo ganz klar, daß"


    # set input_str to command-line argument if given
    input_str = args.inputline
    global REJECTION_WEIGHT
    REJECTION_WEIGHT = args.rejection_weight

    #window_size = 2
    #words_per_window = 3
    #composition_depth = 1
    # TODO: Concatenating two words of the lexicon to one words should be
    # more expensive than the sum of both combined words. Else, merge
    # errors are not corrected.
    #result_num = 10

    if args.punctuation == 'bracket':
        ## bracketing rules
        error_transducer, lexicon_transducer = load_transducers_bracket(
            'fst/max_error_3_context_23_dta.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.Fraktur4.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.deu-frak3.hfst',
            'fst/punctuation_transducer_dta19-reduced.testdata.hfst',
            #'fst/punctuation_transducer_dta19-reduced.traindata.hfst',
            #'fst/lexicon_komplett.again/punctuation_transducer_dta.hfst',
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst',
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst',
            #'fst/lexicon_komplett.again/lexicon_transducer_dta.hfst',
            'fst/open_bracket_transducer_dta19-reduced.testdata.hfst',
            #'fst/open_bracket_transducer_dta19-reduced.traindata.hfst',
            #'fst/lexicon_komplett.again/open_bracket_transducer_dta.hfst',
            'fst/close_bracket_transducer_dta19-reduced.testdata.hfst',
            #'fst/close_bracket_transducer_dta19-reduced.traindata.hfst',
            #'fst/lexicon_komplett.again/close_bracket_transducer_dta.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        #'transducers/morphology_with_identity.hfst')


    elif args.punctuation == 'lm':
        ## inter-word language model
        error_transducer, lexicon_transducer = load_transducers_inter_word(
            'fst/max_error_3_context_23_dta.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.Fraktur4.hfst',
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst',
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst',
            'fst/left_punctuation.hfst',
            'fst/right_punctuation.hfst',
            flag_encoder,
            words_per_window=args.words_per_window,
            composition_depth=args.composition_depth)

    elif args.punctuation == 'preserve':
        ## no punctuation changes
        error_transducer, lexicon_transducer = load_transducers_preserve_punctuation(
            'fst/preserve_punctuation_max_error_3_context_23.hfst',
            #'fst/max_error_3_context_23_preserve_punctuation_dta19-reduced.Fraktur4.hfst',
            'fst/any_punctuation_no_space.hfst',
            'fst/lexicon_transducer_dta19-reduced.testdata.hfst',
            #'fst/lexicon_transducer_dta19-reduced.traindata.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        
    openfst = True # use OpenFST for composition?

    if openfst:

        # write lexicon and error transducer files in OpenFST format
        # (cannot use one file for both with OpenFST::Read)
        with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-error') as error_f:
            with tempfile.NamedTemporaryFile(prefix='cor-asv-fst-sw-lexicon') as lexicon_f:
                write_fst(error_f.name, error_transducer)
                write_fst(lexicon_f.name, lexicon_transducer)
                #write_fst('output/error.fst', error_transducer)
                #write_fst('output/lexicon.fst', lexicon_transducer)
                
                # generate Composition Object
                composition = pyComposition(error_f.name, lexicon_f.name, args.result_num, args.rejection_weight)
                logging.debug(composition)
                
                preparation_done = time.clock()
                logging.info('Preparation Time: %f', preparation_done - start)
                
                # apply correction using Composition Object
                complete_output = window_size_1_2(input_str, None, None, flag_encoder, args.result_num, composition)[0]

    else:

        preparation_done = time.clock()
        logging.info('Preparation Time: %f', preparation_done - start)

        # apply correction directly in hfst

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, args.result_num)[0]

    # write output to filesystem

    #write_fst('output/output_str.nondet.hfst', complete_output)

    complete_output.determinize()

    #write_fst('output/output_str.hfst', complete_output)

    logging.debug('COMPLETE OUTPUT')
    print_output_paths(complete_output)

    #complete_output.n_best(10)

    complete_output = remove_flags(complete_output, flag_encoder)

    #write_fst('output/output_str.without_flags.hfst', complete_output)

    complete_output.output_project()

    logging.debug('COMPLETE OUTPUT NO FLAGS')
    print_shortest_path(complete_output)

    logging.info(list(complete_output.extract_shortest_paths(output='text').items())[0][0].replace(hfst.EPSILON, ''))
    
    ## load and apply language model

    lm_file = 'fst/lang_mod_theta_0_000001.mod.modified.hfst'
    lowercase_file = 'fst/lowercase.hfst'

    lm_fst = helper.load_transducer(lm_file)
    lowercase_fst = helper.load_transducer(lowercase_file)

    complete_output.compose(lowercase_fst)
    
    logging.debug('LOWERCASE OUTPUT')
    print_output_paths(complete_output)

    complete_output.compose(lm_fst)
    complete_output.n_best(10)
    complete_output.input_project() # "undo" lowercase

    logging.debug('LANGUAGE MODEL OUTPUT')
    print_output_paths(complete_output)
    
    logging.info(list(complete_output.extract_shortest_paths(output='text').items())[0][0].replace(hfst.EPSILON, ''))

if __name__ == '__main__':
    main()
