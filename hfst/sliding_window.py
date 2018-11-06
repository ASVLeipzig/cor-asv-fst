import hfst

import time
import math
import string
import sys
import argparse

from composition import pyComposition
import helper

REJECTION_WEIGHT = 1.5 # weight assigned to all transitions in input transducer when disjoining with result transducer as fallback (see set_transition_weights); trade-off between over- and under-correction

def create_input_transducer(input_str):
    """Takes input_str and creates a transducer accepting that string."""

    fst_dict = {input_str: [(input_str, -math.log(1))]}
    input_fst = hfst.fst(fst_dict)

    return input_fst


def prepare_input(input_str, window_size, flag_encoder):
    """Takes long input_string and splits it at space characters into a list of windows with
    window_size words. Between words flags according to the flag_encoder
    are inserted to ensure the correct combination of windows."""

    splitted = input_str.split(' ')

    # combine neighbouring "words", when one of the "words" is merely
    # a single punctuation character

    new_splitted = []

    last_word = ''

    for word in splitted:
        if len(word) == 1 and not word.isalnum():
            if last_word != '':
                last_word = last_word + ' ' + word
            else:
                last_word = word
        elif last_word != '':
            new_splitted.append(last_word)
            last_word = word
        else:
            last_word = word
    new_splitted.append(last_word)

    splitted = new_splitted

    # create input windows

    input_list = []

    for i in range(0, max(1, len(splitted) - window_size + 1)):
        single_input = splitted[i:i + window_size]

        single_input_with_diacritics = flag_encoder.encode(i)

        for j, word in enumerate(single_input):
            single_input_with_diacritics += word + ' '
            single_input_with_diacritics += flag_encoder.encode(i+j+1)

        input_list.append(single_input_with_diacritics)

    #print(input_list)
    return input_list

    # Multichar
    # (for obtaining flags that are single symbols;
    # does not work when using OpenFST, since it removes these flags)

    splitted = input_str.split(' ')

    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]

        single_input_with_diacritics = [flag_encoder.encode(i)]

        for j, word in enumerate(single_input):
            single_input_with_diacritics.append(word)
            single_input_with_diacritics.append(flag_encoder.encode(i+j+1))

        input_list.append(single_input_with_diacritics)

    #print(input_list)
    return input_list


def compose_and_search(input_str, error_transducer, lexicon_transducer, result_num, composition = None):
    """Perform composition and search of result_num best paths for
    input_string. If composition object is given, composition and search
    are executed in OpenFST. Else, HFST is employed.
    The transducer of the input_str with a high wight is disjuncted with
    the result."""

    input_fst = create_input_transducer(input_str)

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

        print('input_str: ', input_str)
        composition.compose(input_str.encode())

        result_fst = helper.load_transducer('output/' + input_str + '.fst')

    else: # compose using HFST

        result_fst = input_fst.copy()

        #print("Input: ", input_str)
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

    result_fst.determinize()
    result_fst.remove_epsilons()
    #result_fst.minimize()

    # disjunct result with input_fst, but with high transition weights
    input_fst = set_transition_weights(input_fst) # acts as a rejection threshold
    result_fst.disjunct(input_fst)

    result_fst.remove_epsilons()

    return result_fst


def set_transition_weights(fst):
    """Sets each transition of the given fst to a high value (10.0)."""

    basic_fst = hfst.HfstBasicTransducer(fst)
    for state in basic_fst.states():
        for transition in basic_fst.transitions(state):
            transition.set_weight(REJECTION_WEIGHT) # 10.0 # hyperparameter!
    return hfst.HfstTransducer(basic_fst)


def print_output_paths(basic_fst):
    """Print the shortest path and five random paths in the basic_fst
    alongside their weight."""

    #complete_paths = hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0)
    complete_paths = hfst.HfstTransducer(basic_fst).extract_shortest_paths()
    for input, outputs in complete_paths.items():
        print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        for output in outputs:
            print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', '□'), output[1]))
    print('Random paths:')
    complete_paths = hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0)
    for input, outputs in complete_paths.items():
        print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        for output in outputs:
            print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', '□'), output[1]))
    print('\n')


def print_shortest_path(basic_fst):
    """Print the shortest path alongside its weight."""

    #complete_paths = hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0)
    complete_paths = hfst.HfstTransducer(basic_fst).extract_shortest_paths()
    for input, outputs in complete_paths.items():
        print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        for output in outputs:
            print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', '□'), output[1]))


def get_flag_states(transducer, starting_state, flag_list):
    """Determine states at the beginning of a flag.
    Additionally, final states are determined to process each state only
    once.
    Only states reachable from starting_state are checked for better
    performance."""

    #print(flag_list)

    flag_state_dict = {} # flag_string -> list of states (e.g. @N.A@ -> [0])
    predecessor_dict = {} # state -> list of predecessor states
    predecessor_dict[0] = []
    final_states = []

    flag_length = len(flag_list[0])
    flag_starting_symbol = flag_list[0][0]

    visited = []

    queue = [starting_state]

    flag_candidates = []

    while queue != []:

        actual_state = queue.pop()
        visited.append(actual_state)

        # remember final states
        if transducer.is_final_state(actual_state):
            final_states.append(actual_state)

        for transition in transducer.transitions(actual_state):

            target_state = transition.get_target_state()

            predecessor_dict[target_state] = predecessor_dict.setdefault(target_state, []) + [actual_state]

            # actual_state is a possible beginning of a flag
            if transition.get_input_symbol() == flag_starting_symbol:
                flag_candidates.append(actual_state)

            # add target states to queue if not visited yet
            if target_state not in visited:
                queue.append(transition.get_target_state())


    #print('candidates', flag_candidates)

    # check for each candidate if the following symbols match a flag
    for candidate in flag_candidates:
        #print(candidate)

        flag_string = ''
        actual_state = candidate
        transition_counter = 0

        while len(flag_string) < flag_length and transition_counter < 8:
            #print(flag_string)
            #print('actual_state', actual_state)
            #print('candidate', candidate)
            for transition in transducer.transitions(actual_state):
                actual_symbol = transition.get_input_symbol()
                if actual_symbol == hfst.EPSILON:
                    actual_symbol = ''
                flag_string += actual_symbol
                actual_state = transition.get_target_state()
                break # process only first transition
            transition_counter += 1

        # if successful, add state to flag_state_dict
        #print(flag_string)
        #print(flag_list)
        if flag_string in flag_list:
            #print('flag found!')
            if flag_state_dict.get(flag_string) == None or candidate not in flag_state_dict.get(flag_string):
                flag_state_dict[flag_string] = flag_state_dict.setdefault(flag_string, []) + [candidate]

        continue

    #print('flag states:', flag_state_dict.items())

    #print('fst', transducer)

    return flag_state_dict, final_states, predecessor_dict


def merge_states(basic_transducer, state_list, predecessor_dict):
    """Merges all states in the given state_list to a single state, the
    first in the list. All incoming and outgoing transitions of the other
    states are redirected over this state.
    Forbid epsilon self-loops."""

    #print('merging following states:', state_list)

    single_state = state_list[0]

    #print('single state', single_state)

    target_states = []
    for state in state_list[1:]:

        #print('from state', state)

        # incoming transitions
        predecessors = predecessor_dict[state]
        for pred in predecessors:
            #print('pred', pred)
            incoming_transitions = []
            for transition in basic_transducer.transitions(pred):
                if transition.get_target_state() == state:
                    #print(transition.get_target_state(), state)
                    incoming_transitions.append((transition.get_target_state(),\
                        transition.get_input_symbol(),\
                        transition.get_output_symbol(),\
                        transition.get_weight()))
                    basic_transducer.remove_transition(pred, transition)
            for (target, input_symbol, output_symbol, weight) in incoming_transitions:
                if pred != single_state:
                    basic_transducer.add_transition(\
                        pred, single_state, input_symbol,\
                        output_symbol, weight)

        # outgoing transitions
        for transition in basic_transducer.transitions(state):
                if single_state != transition.get_target_state():
                    basic_transducer.add_transition(\
                        single_state, transition.get_target_state(), transition.get_input_symbol(),\
                        transition.get_output_symbol(), transition.get_weight())
                    target_states.append(transition.get_target_state())
                    basic_transducer.remove_transition(state, transition)

    # update predecessor_dict
    for state in state_list[1:]:
        predecessor_dict[state] = []
    for state in target_states:
        predecessor_dict[state] = [x for x in predecessor_dict[state] if x not in state_list[1:]]
        if single_state not in predecessor_dict[state]:
            predecessor_dict[state] += [single_state]

    return


def write_fst(name, fst):
    """Write fst to file."""

    fst = hfst.HfstTransducer(fst)

    out = hfst.HfstOutputStream(filename='output/' + name + '.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(fst)
    out.flush()
    out.close()

    return


def combine_results(result_list, window_size, flag_encoder):
    """Takes a list of window results and combines them into a single
    result transducer."""

    #for fst in result_list:
    #    print_output_paths(fst)

    flag_state_dict = {} # flag_string -> list of states (e.g. @N.A@ -> [0])
    final_states = []

    # start with first window
    starting_fst = result_list[0].copy()
    #starting_fst.output_project()
    #starting_fst.minimize()
    #starting_fst.remove_epsilons()

    write_fst('starting_fst', starting_fst)

    result_fst = hfst.HfstBasicTransducer(starting_fst)
    
    starting_fst.output_project()
    print('WINDOW RESULT PATHS')
    print_shortest_path(starting_fst)
    
    flag_state_dict, final_states, predecessor_dict = get_flag_states(result_fst, 0, flag_encoder.flag_list)

    #print('flag states', flag_state_dict.items())
    #print('final states', final_states)


    # merge states in initial transducer
    merge_list = flag_state_dict[flag_encoder.encode(1)]
    merge_states(result_fst, merge_list, predecessor_dict)
    # update flag_state_dict
    flag_state_dict[flag_encoder.encode(1)] = [merge_list[0]]


    #print('PARTIAL RESULT PATHS')
    #print_output_paths(starting_fst)


    for i, fst in enumerate(result_list[1:]):

        #if i == 1:
        #    sys.exit(0)

        fst.output_project()
        #fst.minimize()
        #fst.remove_epsilons()

        #write_fst('partial_result', fst)

        partial_fst = hfst.HfstBasicTransducer(fst)

        print('WINDOW RESULT PATHS')
        print_shortest_path(partial_fst)

        #  remove final states in result fst
        for state in final_states:
            result_fst.remove_final_weight(state)

        # determine append states and set them final
        append_flag = flag_encoder.encode(i+1)
        append_states = flag_state_dict[append_flag]
        for state in append_states:
            result_fst.set_final_weight(state, 0.0)

        #print('append_states', append_states)

        #print('BEFORE CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)
        #write_fst('before_concatenation', result_fst)

        # concatenate result fst and partial result
        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)

        result_fst.concatenate(partial_fst)

        ##print("number of states :", result_fst.number_of_states())
        #print('PARTIAL RESULT PATHS')
        #print_output_paths(partial_fst)
        #print('AFTER CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)
        ##result_fst.n_best(100)

        result_fst = hfst.HfstBasicTransducer(result_fst)

        # update final_states and flag_state_dict
        flag_state_dict, final_states, predecessor_dict = get_flag_states(result_fst, 0, flag_encoder.flag_list)

        #print('before merge')
        #print('flag states', flag_state_dict.items())
        #print('final states', final_states)

        #write_fst('before_merge', result_fst)
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
        #write_fst('after_merge', result_fst)

        #print('after merge')
        #print('flag states', flag_state_dict.items())
        #print('final states', final_states)

        #continue



        ## merge word borders
        ## merge_word_borders(result_fst, path_dict, predecessor_dict)

        ##print('AFTER MERGE RESULT PATHS')
        ##print_output_paths(result_fst)


        ##print('before merge', result_fst)
        #write_fst('before_merge', result_fst)

        ## merge corresponding states
        #merge_list = flag_state_dict[flag_encoder.encode(i+1)]
        #merge_states(result_fst, merge_list, predecessor_dict)

        ## update flag_state_dict
        #flag_state_dict[flag_encoder.encode(i+1)] = [merge_list[0]]

        ##print('after merge', result_fst)
        #write_fst('after_merge', result_fst)

    #sys.exit(0)

    return result_fst


def create_result_transducer(input_str, window_size, words_per_window, error_transducer, lexicon_transducer, result_num, flag_encoder, composition=None):
    """Prepares the input_str for a given window_size and performs the
    composition and search of result_num best paths on each of the windows.
    The window results are combined to a single transducer. """

    #lexicon_transducer.repeat_n(words_per_window)

    start = time.time()

    input_list = prepare_input(input_str, window_size, flag_encoder)
    #print("Input List: ", input_list)
    output_list = []

    for i, single_input in enumerate(input_list):
        #print("Single Input: ", single_input)

        results = compose_and_search(single_input, error_transducer, lexicon_transducer, result_num, composition)

        output_list.append(results)

    after_composition = time.time()

    #complete_output = combine_results(output_list, window_size)
    #complete_output = remove_redundant_paths(complete_output)
    
    complete_output = combine_results(output_list, window_size, flag_encoder)
    complete_output = hfst.HfstTransducer(complete_output)
    
    after_combination = time.time()

    print('Composition Time: ', after_composition - start)
    print('Combination Time: ', after_combination - after_composition)

    return complete_output


def get_edit_space_transducer(flag_encoder):
    """Reads a space and a flag diacritic for marking word borders and
    removes it. Needed to replace space to epsilon edits in the error
    transducer to handle merges of two words."""

    remove_space_transducer = hfst.regex('% :0')

    flag_list = []
    for i in range(flag_encoder.first_input_int, flag_encoder.max_input_int + 1):
        flag_list.append(flag_encoder.encode(i))

    remove_diacritics_transducer = hfst.HfstBasicTransducer()
    tok = hfst.HfstTokenizer()
    for flag in flag_list:
        remove_diacritics_transducer.disjunct(tok.tokenize(flag, ''), 0.0)

    #print(remove_diacritics_transducer)
    remove_diacritics_transducer = hfst.HfstTransducer(remove_diacritics_transducer)
    remove_diacritics_transducer.optionalize()

    remove_space_transducer.concatenate(remove_diacritics_transducer)
    remove_space_transducer.minimize()

    #print(remove_space_transducer)

    return remove_space_transducer


def get_flag_acceptor(flag_encoder):
    """Transducer that accepts the flags of the flag encoder. It is needed
    before and after each concatenated lexicon transducer to mark the word
    borders."""

    flag_list = []
    for i in range(flag_encoder.first_input_int, flag_encoder.max_input_int + 1):
        flag_list.append(flag_encoder.encode(i))

    flag_acceptor = hfst.HfstBasicTransducer()
    tok = hfst.HfstTokenizer()
    for flag in flag_list:
        flag_acceptor.disjunct(tok.tokenize(flag), 0.0)

    #print('Flag Acceptor: ', flag_acceptor)
    flag_acceptor = hfst.HfstTransducer(flag_acceptor)
    flag_acceptor.optionalize()

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


def load_transducers_bracket(error_file,
    punctuation_file,
    lexicon_file,
    open_bracket_file,
    close_bracket_file,
    flag_encoder,
    composition_depth=1,
    words_per_window=3,
    morphology_file=None):
    """Load transducers for using the bracket model, including the
    punctuation fst, lexicon fst, and opening/closing bracket fst.
    Concatenate the transducers to a single fst with words_per_window
    concatenations of the lexicon."""

    # load transducers

    flag_acceptor = get_flag_acceptor(flag_encoder)

    error_transducer = helper.load_transducer(error_file)
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))

    punctuation_transducer = helper.load_transducer(punctuation_file)
    punctuation_transducer.optionalize()

    open_bracket_transducer = helper.load_transducer(open_bracket_file)
    open_bracket_transducer.optionalize()
    close_bracket_transducer = helper.load_transducer(close_bracket_file)
    close_bracket_transducer.optionalize()

    lexicon_transducer = helper.load_transducer(lexicon_file)

    space_transducer = hfst.regex('% :% ')

    # add morphology to lexicon

    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)

    # add composed words to lexicon

    if composition_depth > 1:

        connect_composition = hfst.regex('s:s')
        connect_composition.optionalize()

        optional_lexicon_transducer = lexicon_transducer.copy()
        optional_lexicon_transducer.optionalize()

        connect_composition.concatenate(optional_lexicon_transducer)
        connect_composition.repeat_n(composition_depth - 1)

        lexicon_transducer.concatenate(connect_composition)

    # combine transducers to lexicon transducer
    result_lexicon_transducer = flag_acceptor.copy()
    result_lexicon_transducer.concatenate(open_bracket_transducer)
    result_lexicon_transducer.concatenate(lexicon_transducer)
    result_lexicon_transducer.concatenate(punctuation_transducer)
    result_lexicon_transducer.concatenate(close_bracket_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    result_lexicon_transducer.optionalize()

    result_lexicon_transducer.repeat_n(words_per_window)

    result_lexicon_transducer.concatenate(flag_acceptor)

    return error_transducer, result_lexicon_transducer



def load_transducers_preserve_punctuation(error_file,
    punctuation_file,
    lexicon_file,
    flag_encoder,
    composition_depth=1,
    words_per_window=3,
    morphology_file=None):
    """Load transducers for preserving punctuation, including the
    punctuation fst (for any punctuation) and lexicon fst.
    The error model should be unable to change any characters into punctuation
    characters.
    Concatenate the transducers to a single fst with words_per_window
    concatenations of the lexicon."""

    # load transducers

    flag_acceptor = get_flag_acceptor(flag_encoder)

    error_transducer = helper.load_transducer(error_file)
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))

    punctuation_transducer = helper.load_transducer(punctuation_file)
    #punctuation_transducer.optionalize()

    #open_bracket_transducer = helper.load_transducer(open_bracket_file)
    #open_bracket_transducer.optionalize()
    #close_bracket_transducer = helper.load_transducer(close_bracket_file)
    #close_bracket_transducer.optionalize()

    lexicon_transducer = helper.load_transducer(lexicon_file)

    space_transducer = hfst.regex('% :% ')

    # add morphology to lexicon

    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)

    # add composed words to lexicon

    if composition_depth > 1:

        connect_composition = hfst.regex('s:s')
        connect_composition.optionalize()

        optional_lexicon_transducer = lexicon_transducer.copy()
        optional_lexicon_transducer.optionalize()

        connect_composition.concatenate(optional_lexicon_transducer)
        connect_composition.repeat_n(composition_depth - 1)

        lexicon_transducer.concatenate(connect_composition)

    # combine transducers to lexicon transducer
    result_lexicon_transducer = flag_acceptor.copy()
    #result_lexicon_transducer.concatenate(open_bracket_transducer)
    result_lexicon_transducer.concatenate(punctuation_transducer)
    result_lexicon_transducer.concatenate(lexicon_transducer)
    result_lexicon_transducer.concatenate(punctuation_transducer)
    #result_lexicon_transducer.concatenate(close_bracket_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    result_lexicon_transducer.optionalize()

    result_lexicon_transducer.repeat_n(words_per_window)

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
    """Load transducers for using the inter-word model, including the
    lexicon fst, and left/right punctuation fst.
    Concatenate the transducers to a single fst with words_per_window
    concatenations of the lexicon."""

    # TODO: handle flag diacritics; the construction of the lexicon is not
    # correct anymore; the punctuation_right_transducer as constructed
    # contains a space character at the beginning which shouldn't be there,
    # since the space is placed at the end of each window (with no
    # punctuation characters after that); since this model should be
    # changed to a complete punctuation ngram model anyway, the corrections
    # can be made on that occasion

    # load transducers

    flag_acceptor = get_flag_acceptor(flag_encoder)
    
    error_transducer = helper.load_transducer(error_file)

    punctuation_left_transducer = helper.load_transducer(punctuation_left_file)
    punctuation_right_transducer = helper.load_transducer(punctuation_right_file)

    punctuation_left_transducer.optionalize()
    punctuation_right_transducer.optionalize()

    lexicon_transducer = helper.load_transducer(lexicon_file)

    space_transducer = hfst.regex('% :% ')

    # add morphology to lexicon

    if morphology_file != None:
        morphology_transducer = helper.load_transducer(morphology_file)
        lexicon_transducer.compose(morphology_transducer)

    # add composed words to lexicon

    if composition_depth > 1:

        connect_composition = hfst.regex('s:s')
        connect_composition.optionalize()

        optional_lexicon_transducer = lexicon_transducer.copy()
        optional_lexicon_transducer.optionalize()

        connect_composition.concatenate(optional_lexicon_transducer)
        connect_composition.repeat_n(composition_depth - 1)

        lexicon_transducer.concatenate(connect_composition)


    # combine transducers to lexicon transducer

    result_lexicon_transducer = lexicon_transducer.copy()
    result_lexicon_transducer.concatenate(punctuation_left_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    result_lexicon_transducer.concatenate(punctuation_right_transducer)
    result_lexicon_transducer.optionalize()

    #result_lexicon_transducer.repeat_n(words_per_window)
    result_lexicon_transducer.repeat_n(3)

    output_lexicon = punctuation_right_transducer.copy()
    output_lexicon.concatenate(result_lexicon_transducer)

    final_result_lexicon_transducer = flag_acceptor.copy()
    final_result_lexicon_transducer.concatenate(output_lexicon)
    final_result_lexicon_transducer.concatenate(flag_acceptor)

    return error_transducer, output_lexicon


def complete_merge(basic_fst, flag_encoder):
    """Merge all states that are predecessors of the special merge
    flags. This has to be performed when windows of size 1 and 2 are
    combined."""

    flag_state_dict, final_states, predecessor_state = get_flag_states(basic_fst, 0, flag_encoder.flag_list)

    for flag in flag_state_dict.keys():
        flag_states = flag_state_dict[flag]
        merge_states(basic_fst, flag_states, predecessor_state)

    merge_states(basic_fst, final_states, predecessor_state)

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

    window_1 = create_result_transducer(\
        input_str, 1, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    window_2 = create_result_transducer(\
        input_str, 2, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    window_1.disjunct(window_2)
    complete_output_basic = hfst.HfstBasicTransducer(window_1)

    before_merge = time.time()

    complete_merge(complete_output_basic, flag_encoder)

    after_merge = time.time()
    print('Merge Time: ', after_merge - before_merge)

    complete_output = hfst.HfstTransducer(complete_output_basic)

    #complete_output.prune()
    #complete_output.minimize()

    return complete_output


def remove_flags(fst, flag_encoder):
    """Removes flags for construction of sliding window from a basic
    fst, given the corresponding flag_encoder. This is performed by
    deleting the transitions leaving the flag states (at the beginning of a
    flag) and adding an epsilon transition to the state after the flag.
    """

    flag_state_dict, final_states, predecessor_dict = get_flag_states(fst, 0, flag_encoder.flag_list)

    flag_length = len(flag_encoder.flag_list[0])

    remove_transitions = []
    add_transitions = []

    flag_states = []
    for key in flag_state_dict.keys():
        flag_states += flag_state_dict[key]
    #print(flag_states)

    for state in flag_states:
        for transition in fst.transitions(state):
            new_target_state = transition.get_target_state()

            remove_transitions.append((state, new_target_state, transition.get_input_symbol(),\
                transition.get_output_symbol(), transition.get_weight()))
            #((state, transition.get_target_state(), transition.get_input_symbol(),\
            #    transition.get_output_symbol(), transition.get_weight())

            for i in range(0, flag_length - 1):
                for inner_transition in fst.transitions(new_target_state):
                    new_target_state = inner_transition.get_target_state()
                    break

            add_transitions.append((state,\
                hfst.HfstBasicTransition(new_target_state, hfst.EPSILON,\
                hfst.EPSILON, transition.get_weight())))

    for entry in remove_transitions:
        fst.remove_transition(entry[0], hfst.HfstBasicTransition(entry[1], entry[2], entry[3], entry[4]))

    for entry in add_transitions:
        fst.add_transition(entry[0], entry[1])

    return fst


class FlagEncoder:
    """Endode and decode integers (between first_input_int and
    max_input_int) to flags used in the transducer.
    All flags are required to have the same length, else this will cause
    problems in the functions get_flag_states and remove_flags.
    """
    # TODO: allow a wider range of integers to encode, for example by using
    # two consecutive alphabetical characters (26*26 = 676)
    # TODO: check/ensure that the flag cannot be modified by the
    # error_transducer randomly

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
        for i in range(self.first_input_int, self.max_input_int + 1):
            self.flag_list.append(self.encode(i))


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

    start = time.time()

    # command line options
    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst one-shot tool')
    parser.add_argument('inputline', metavar='STRING', default=u"Philoſophenvon Fndicn dur<h Gricche nland bis", help='specify input string')
    parser.add_argument('-P', '--punctuation', metavar='MODEL', type=str, choices=['bracket', 'lm', 'preserve'], default='bracket', help='how to model punctuation between words (bracketing rules, inter-word language model, or keep unchanged)')
    parser.add_argument('-W', '--words-per-window', metavar='WORDS', type=int, default=3, help='maximum number of words in one window')
    parser.add_argument('-R', '--result-num', metavar='RESULTS', type=int, default=10, help='result paths per window')
    parser.add_argument('-D', '--composition-depth', metavar='DEPTH', type=int, default=1, help='number of lexicon words that can be concatenated')
    args = parser.parse_args()

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
            'fst/punctuation_transducer_dta.hfst',
            'fst/lexicon_transducer_dta.hfst',
            'fst/open_bracket_transducer_dta.hfst',
            'fst/close_bracket_transducer_dta.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        #'transducers/morphology_with_identity.hfst')


    elif args.punctuation == 'lm':
        ## inter-word language model
        error_transducer, lexicon_transducer = load_transducers_inter_word(
            'fst/max_error_3_context_23_dta.hfst',
            #'fst/max_error_3_context_23_dta19-reduced.Fraktur4.hfst',
            'fst/lexicon_transducer_dta.hfst',
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
            'fst/lexicon_transducer_dta.hfst',
            flag_encoder,
            composition_depth=args.composition_depth,
            words_per_window=args.words_per_window)
        

    openfst = True # use OpenFST for composition?

    if openfst:

        # write lexicon and error transducer in OpenFST format

        error_filename = u'error.ofst'
        lexicon_filename = u'lexicon.ofst'

        for filename, fst in [(error_filename, error_transducer), (lexicon_filename, lexicon_transducer)]:
            out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            out.write(fst)
            out.flush()
            out.close()

        # generate Composition Object

        composition = pyComposition(error_filename.encode('utf-8'), lexicon_filename.encode('utf-8'), args.result_num)
        print(composition)

        preparation_done = time.time()
        print('Preparation Time: ', preparation_done - start)

        # apply correction using Composition Object

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, args.result_num, composition)


    else:

        preparation_done = time.time()
        print('Preparation Time: ', preparation_done - start)

        # apply correction directly in hfst

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, args.result_num)

    # write output to filesystem

    #complete_output.prune()
    complete_output.determinize()

    out = hfst.HfstOutputStream(filename='output/' + input_str + '.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(complete_output)
    out.flush()
    out.close()

    print('COMPLETE OUTPUT')
    print_output_paths(complete_output)

    #complete_output.n_best(10)

    complete_output = remove_flags(hfst.HfstBasicTransducer(complete_output), flag_encoder)
    complete_output = hfst.HfstTransducer(complete_output)

    out = hfst.HfstOutputStream(filename='output/' + input_str + '_removed_flags.hfst',
        hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(complete_output)
    out.flush()
    out.close()

    print('COMPLETE OUTPUT NO FLAGS')
    print_output_paths(complete_output)

    ## load and apply language model

    lm_file = 'fst/lang_mod_theta_0_000001.mod.modified.hfst'
    lowercase_file = 'fst/lowercase.hfst'

    lm_fst = helper.load_transducer(lm_file)
    lowercase_fst = helper.load_transducer(lowercase_file)

    complete_output.output_project()

    complete_output.compose(lowercase_fst)

    print('Lowercase Output')
    print_output_paths(complete_output)

    complete_output.compose(lm_fst)

    complete_output.n_best(10)

    print('Language Model Output')
    print_output_paths(complete_output)


if __name__ == '__main__':
    main()
