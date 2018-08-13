import hfst
import libhfst

import time
import math
import string

import error_transducer as et

from composition import pyComposition



#def remove_redundant_paths(basic_fst):
#
#    initial_state = 0
#
#    # key: state_id, value: tuple containing three dicts
#    # predecessor -> output_string, output_string -> predecessor,
#    # output_string -> weight, string -> transition
#    path_dict = {}
#
#    states = basic_fst.states()
#    num_states = len(states)
#    visited = []
#    queue = [initial_state]
#
#    #path_dict[initial_state] = [('', 0.0, None)]
#    path_dict[initial_state] = ({-1: ['']}, {'': -1}, {'': 0.0}, {'': None}, {None: ['']})
#
#
#    #while len(visited) < num_states:
#    while queue != []:
#
#        actual_state = queue.pop()
#        visited.append(actual_state)
#
#        actual_pre_strings_dict = path_dict[actual_state][0]
#        actual_string_pre_dict = path_dict[actual_state][1]
#        actual_string_weight_dict = path_dict[actual_state][2]
#        actual_string_transition_dict = path_dict[actual_state][3]
#        actual_transition_strings_dict = path_dict[actual_state][4]
#
#        #print()
#        #print('Actual State: ', actual_state)
#        #print('Queue: ', queue)
#        #print('Visited: ', visited)
#
#        # iterate over outgoing transitions
#        transitions = basic_fst.transitions(actual_state)
#        for transition in transitions:
#
#            target = transition.get_target_state()
#            #print()
#            #print('Target: ', target)
#
#            transition_string = str(actual_state) + ' ' + transition.__str__()
#            #print('Transition: ', transition_string)
#
#            strings_improved = []
#            strings_worsened = []
#            strings_equal = []
#
#
#            target_pre_strings_dict = {}
#            target_string_pre_dict = {}
#            target_string_weight_dict = {}
#            target_string_transition_dict = {}
#            target_transition_strings_dict = {}
#
#            #print('Path Dict Keys: ', path_dict.keys())
#            target_dicts = path_dict.get(target)
#
#            if target_dicts != None:
#
#                #print('Target Dict already set.')
#
#                target_pre_strings_dict = target_dicts[0]
#                target_string_pre_dict = target_dicts[1]
#                target_string_weight_dict = target_dicts[2]
#                target_string_transition_dict = target_dicts[3]
#                target_transition_strings_dict = target_dicts[4]
#
#
#            for output_string in list(actual_string_pre_dict.keys()):
#
#                #print('String at predecessor: ', output_string)
#
#                new_output_symbol = transition.get_output_symbol()
#                if new_output_symbol == hfst.EPSILON:
#                    new_output_symbol = ''
#                new_output_string = output_string + new_output_symbol
#                #print('New String: ', new_output_string)
#                new_weight = actual_string_weight_dict[output_string] + transition.get_weight()
#
#                old_weight = target_string_weight_dict.get(new_output_string)
#                if old_weight != None:
#                    if old_weight < new_weight:
#                        strings_worsened.append(new_output_string)
#                    elif old_weight == new_weight:
#                        strings_equal.append(new_output_string)
#                    else:
#                        strings_improved.append(new_output_string)
#
#                else:
#                    strings_improved.append(new_output_string)
#
#            # if the new transition creates no new or better
#            # output_strings, remove transition
#            if strings_improved == []:
#                old_transitions = list(target_string_transition_dict.values())
#
#                splitted_transition = transition_string.split()
#                splitted_old_transitions = list(map(lambda x: x.split(), old_transitions))
#
#                equivalent_transition = [x for x in splitted_old_transitions if\
#                    len(x) == 5 and len(splitted_transition) == 5 and\
#                    x[0] == splitted_transition[0] and\
#                    x[1] == splitted_transition[1] and\
#                    x[2] == splitted_transition[2] and\
#                    x[3] == splitted_transition[3]]
#
#                if transition_string not in old_transitions:
#                    basic_fst.remove_transition(actual_state, transition)
#                    #print('Transition led to no improvement! Remove new transition: ',\
#                    #    actual_state, transition)
#                    #print('Old transitions: ', list(target_string_transition_dict.values()))
#
#                    # add equivalent transition (same starting state),
#                    # because it gets removed with remove_transition
#                    # (ignored different weights)
#                    if equivalent_transition != []:
#                        good_old_transition = equivalent_transition[0]
#                        #splitted = good_old_transition.split()
#                        splitted = good_old_transition
#                        basic_fst.add_transition(int(splitted[0]),\
#                            hfst.HfstBasicTransition(int(splitted[1]), splitted[2], splitted[3], float(splitted[4])))
#
#            else:
#
#                #print('Improved Strings: ', strings_improved)
#                #print('Worsened Strings: ', strings_worsened)
#                #print('Equal Strings: ', strings_equal)
#
#
#                removal_candidates = []
#                new_candidates = []
#
#                for string in strings_improved:
#                    pred = target_string_pre_dict.get(string)
#                    if pred != None:
#                        removal_candidates.append(string)
#                    else:
#                        new_candidates.append(string)
#                for string in strings_equal:
#                    pred = target_string_pre_dict.get(string)
#                    if pred != None:
#                        removal_candidates.append(string)
#                    else:
#                        new_candidates.append(string)
#
#
#                #print('Removal Candidates: ', removal_candidates)
#
#                removal_transitions = []
#
#                for removal_candidate in removal_candidates:
#                    old_predecessor = target_string_pre_dict.get(removal_candidate)
#                    old_transition = target_string_transition_dict.get(removal_candidate)
#                    old_transition_strings = target_transition_strings_dict.setdefault(removal_candidate, [])
#
#                    remove = True
#
#                    for tr in old_transition_strings:
#                        if tr not in removal_candidates:
#                            remove = False
#
#                    if remove:
#                        removal_transitions.append((old_predecessor, old_transition))
#
#
#                # remove redundant transitions
#
#                for st, tr in removal_transitions:
#                    if tr != transition_string:
#                        splitted = transition_string.split()
#                        #print('Transition String: ', transition_string)
#                        if len(splitted) != 5:
#                            break
#                        basic_fst.remove_transition(\
#                            st, hfst.HfstBasicTransition(\
#                            int(splitted[1]), splitted[2], splitted[3], float(splitted[4])))
#                        #print('Remove old redundant transition: ', st, tr)
#                        #print('New better transition: ', print(transition_string))
#
#                        # add new better transition, if it has same
#                        # starting state, because remove_transition removes
#                        # it ignoring different weight
#                        if st == int(splitted[0]):
#                            basic_fst.add_transition(actual_state, transition)
#
#                # update dicts
#
#                #all_candidates = removal_candidates + new_candidates
#                all_candidates = strings_improved + strings_equal
#                #print('All candidates: ', all_candidates)
#
#                target_pre_strings_dict[actual_state] = all_candidates
#
#                for string in all_candidates:
#                    target_string_pre_dict[string] = actual_state
#
#                    #new_weight = actual_string_weight_dict[string[0:-1]] + transition.get_weight()
#                    #print(transition_string)
#                    #print(actual_string_weight_dict.keys())
#                    #if transition_string.split()[3] == '@_EPSILON_SYMBOL_@':
#                    if transition.get_output_symbol() == hfst.EPSILON:
#                        new_weight = actual_string_weight_dict[string] + transition.get_weight()
#                    else:
#                        new_weight = actual_string_weight_dict[string[0:-1]] + transition.get_weight()
#                    #print('Predecessor weight: ', actual_string_weight_dict[string[0:-1]])
#                    #print('Target weight: ', new_weight)
#                    target_string_weight_dict[string] = new_weight
#
#                    target_string_transition_dict[string] = transition_string
#
#                target_transition_strings_dict[transition_string] = all_candidates
#
#
#                new_target_dicts =\
#                    (target_pre_strings_dict, target_string_pre_dict,\
#                    target_string_weight_dict, target_string_transition_dict,\
#                    target_transition_strings_dict)
#
#                path_dict[target] = new_target_dicts
#                #print('String-Weight Keys: ', path_dict[target][2].keys())
#
#                # add target to queue
#
#                if target not in queue:
#                    queue.append(target)
#
#    return basic_fst



def get_path_dicts(basic_fst):

    initial_state = 0

    path_dict = {}          # input string leading to state
    output_dict = {}        # output string leading to state
    transition_dict = {}    # transition leading to state
    predecessor_dict = {}   # predecessor state of state

    states = basic_fst.states()
    num_states = len(states)
    visited = []
    queue = [initial_state]

    while len(visited) < num_states:

        actual_state = queue.pop()
        visited.append(actual_state)

        # set string leading to actual_state
        if len(visited) > 1:
            predecessor = predecessor_dict[actual_state]
            input_symbol = transition_dict[actual_state].get_input_symbol()
            if input_symbol == hfst.EPSILON:
                input_symbol = ''
            path_dict[actual_state] = path_dict[predecessor] + input_symbol
            output_symbol = transition_dict[actual_state].get_output_symbol()
            if output_symbol == hfst.EPSILON:
                output_symbol = ''
            output_dict[actual_state] = output_dict[predecessor] + output_symbol
        else:
            path_dict[actual_state] = ''
            output_dict[actual_state] = ''

        # add new states to queue
        transitions = basic_fst.transitions(actual_state)
        for transition in transitions:
            target = transition.get_target_state()
            if target not in visited and target not in queue:
                predecessor_dict[target] = actual_state
                transition_dict[target] = transition
                queue.append(target)

    return path_dict, output_dict, transition_dict, predecessor_dict


def merge_word_borders(basic_fst, path_dict, predecessor_dict):

   # find states with space as input symbol and output_symbol
    input_space_states = []
    for state, transitions in enumerate(basic_fst):
        for transition in transitions:
            if transition.get_input_symbol() == ' ' and transition.get_output_symbol() == ' ':
            #if transition.get_input_symbol() == ' ':
                input_space_states.append(transition.get_target_state())

    # states should merge before the space
    merge_candidates = []
    for state in input_space_states:
        predecessor = predecessor_dict[state]
        #print("predecessor path: ", path_dict[predecessor])
        #if path_dict[predecessor][-1] == ' ':
        #    print("ALARM")
        merge_candidates.append(predecessor)

    # add states without transitions/successors to merge candidates
    for state in basic_fst.states():
        if basic_fst.transitions(state) == ():
            #print('added state ', state, ' to merge candidates')
            merge_candidates.append(state)

    #print("merge candidates: ", merge_candidates)

    # find pairs of corresponding states and merge
    # compare with other merge_candidates, because new last states are
    # handled by adding states without outgoing transitions
    for i, state1 in enumerate(merge_candidates):
        for state2 in merge_candidates[i:]:
            if path_dict[state1] == path_dict[state2]:
                #print("corresponding input string ", path_dict[state1])
                #print("output_string1 ", output_dict[state1])
                #print("output_string2 ", output_dict[state2])
                transition1_exists = False
                transition2_exists = False

                for transition in basic_fst.transitions(state1):
                    if transition.get_target_state() == state2 \
                    and transition.get_input_symbol() == hfst.EPSILON \
                    and transition.get_output_symbol() == hfst.EPSILON:
                        transition1_exists = True

                for transition in basic_fst.transitions(state2):
                    if transition.get_target_state() == state1 \
                    and transition.get_input_symbol() == hfst.EPSILON \
                    and transition.get_output_symbol() == hfst.EPSILON:
                        transition2_exists = True

                if not transition1_exists:
                    #print("found corresponding states with path ", path_dict[state1])
                    basic_fst.add_transition(state1, state2, hfst.EPSILON, hfst.EPSILON, 0.0)

                if not transition2_exists:
                    basic_fst.add_transition(state2, state1, hfst.EPSILON, hfst.EPSILON, 0.0)

    #return output_dict, path_dict


def create_input_transducer(input_str):

    fst_dict = {input_str: [(input_str, -math.log(1))]}
    input_fst = hfst.fst(fst_dict)

    return input_fst



    # Triple Input

    #(input_diacritic, input_str, output_diacritic) = input_triple

    #input_diacritic_fst = hfst.regex('"' + input_diacritic + '"')
    #output_diacritic_fst = hfst.regex('"' + output_diacritic + '"')

    #fst_dict = {input_str: [(input_str, -math.log(1))]}
    #input_fst = hfst.fst(fst_dict)


    #complete_input_fst = input_diacritic_fst.copy()
    #complete_input_fst.concatenate(input_fst)
    #complete_input_fst.concatenate(output_diacritic_fst)

    #return complete_input_fst


    # Multi-Char

    #fst_to_concatenate = []

    #for i, element in enumerate(input_list):
    #    if i % 2 == 0:
    #        # append diacritical symbol as it is
    #        regex = '"' + element + '":"' + element + '"::0.0'
    #        d_fst = hfst.regex(regex)
    #        fst_to_concatenate.append(d_fst)
    #    else:
    #        # append space character after word
    #        element += ' '
    #        fst_dict = {element: [(element, -math.log(1))]}
    #        w_fst = hfst.fst(fst_dict)
    #        fst_to_concatenate.append(w_fst)

    #input_fst = fst_to_concatenate[0].copy()

    #for fst in fst_to_concatenate[1:]:
    #    input_fst.concatenate(fst)

    #print('#INPUT FST')
    #print(i#nput_fst)

    #return input_fst




def prepare_input(input_str, window_size, flag_encoder):
    """Takes long input_string and splits it at space characters into a list of windows with
    window_size words."""

    #splitted = input_str.split(' ')

    #new_splitted = []

    #for i, word in enumerate(splitted):
    #    word = flag_encoder.encode(i) + word
    #    new_splitted.append(word)

    #splitted = new_splitted

    ##if len(splitted) < window_size:
    ##    return ' '.join(splitted) + ' '

    #input_list = []

    #for i in range(0, len(splitted) - window_size + 1):
    #    single_input = splitted[i:i + window_size]
    #    input_list.append(' '.join(single_input) + ' ' + flag_encoder.encode(i + window_size))

    #print(input_list)
    #return input_list

    #new_splitted = []

    #for i, word in enumerate(splitted):
    #    word = flag_encoder.encode(i) + word
    #    new_splitted.append(word)

    #splitted = new_splitted




    splitted = input_str.split(' ')

    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]

        single_input_with_diacritics = flag_encoder.encode(i)

        for j, word in enumerate(single_input):
            single_input_with_diacritics += word + ' '
            single_input_with_diacritics += flag_encoder.encode(i+j+1)

        input_list.append(single_input_with_diacritics)

    print(input_list)
    return input_list



    # Multi-Char

    splitted = input_str.split(' ')

    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]

        single_input_with_diacritics = [flag_encoder.encode(i)]

        for j, word in enumerate(single_input):
            single_input_with_diacritics.append(word)
            single_input_with_diacritics.append(flag_encoder.encode(i+j+1))

        input_list.append(single_input_with_diacritics)

    print(input_list)
    return input_list


def input_list_to_str(input_list, with_diacritics=False):

    output_str = ''

    for i, element in enumerate(input_list):
        if i % 2 == 1:
            output_str += element + ' '
        elif with_diacritics:
            output_str += element

    return output_str


def compose_and_search(input_str, error_transducer, lexicon_transducer, result_num, composition = None):

    input_fst = create_input_transducer(input_str)

    # TODO: OpenFST-Behandlung der Diacritics/Liste
    if composition != None:

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

        result_fst = et.load_transducer('output/' + input_str + '.fst')


    else:

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

    input_fst = set_transition_weights(input_fst)

    #if result_fst.number_of_states() == 0 or result_fst.number_of_arcs() == 0:
    #    #print("result empty")
    #    #input_fst.set_final_weights(100.0)
    #    input_fst = set_transition_weights(input_fst)
    #    return input_fst

    #print('RESULT FST')
    #print(result_fst)

    #else:
    result_fst.disjunct(input_fst)


    return result_fst


def set_transition_weights(fst):
    basic_fst = hfst.HfstBasicTransducer(fst)
    for state in basic_fst.states():
        for transition in basic_fst.transitions(state):
            transition.set_weight(10.0)
    return hfst.HfstTransducer(basic_fst)


def get_append_states(basic_fst, window_size, path_dict, output_dict):

    # take all unique input strings, sort by length, filter for strings
    # ending in space
    input_strings = path_dict.values()
    input_strings = list(set(input_strings))
    input_strings = list(filter(lambda x: len(x) > 0, input_strings))
    input_strings = list(filter(lambda x: x[-1] == ' ', input_strings))
    input_strings.sort(key=len)

    # take input string with correct position depending on window_size
    append_string = input_strings[- window_size]

    #print('APPEND STRING: ', append_string)

    # take all states with the correct input string
    items = list(path_dict.items())
    filtered_items = list(filter(lambda x: x[1] == append_string, items))

    append_state_candidates = list(map(lambda x: x[0], filtered_items))
    #print('APPEND STATE CANDIDATES: ', append_state_candidates)

    # filter for states with ingoing transition with output space
    #append_states = []
    #for state, transitions in enumerate(basic_fst):
    #    for transition in transitions:
    #        if transition.get_output_symbol in ' '\
    #        if transition.get_target_state() in append_state_candidates:
    #        #and transition.get_target_state() not in append_states:
    #            print("candidate found")
    #            print("input character ", transition.get_output_symbol())
    #            append_states.append(transition.get_target_state())

    #print('Output Dict: ', list(map(lambda x: output_dict[x], append_state_candidates)))

    append_states = list(filter(lambda x: output_dict[x] == '' or output_dict[x][-1] == ' ', append_state_candidates))

    #print('APPEND STATES: ', append_states)

    return append_states, append_state_candidates


def print_output_paths(basic_fst):
    #complete_paths = hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0)
    complete_paths = hfst.HfstTransducer(basic_fst).extract_shortest_paths()
    for input, outputs in complete_paths.items():
        print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        for output in outputs:
            print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))
    print('\n')





def get_flag_states(transducer, starting_state):

    return


def combine_results_new(result_list, window_size):
    """Takes a list of window results and combines them into a single
    result transducer."""

    flag_state_dict = {}
    final_states = {}

    # start with first window
    starting_fst = result_list[0].copy()
    result_fst = hfst.HfstBasicTransducer(starting_fst)

    # merge word borders
    # merge_word_borders(result_fst, path_dict, predecessor_dict)

    for i, fst in enumerate(result_list[1:]):

        partial_fst = hfst.HfstBasicTransducer(fst)

        #  remove final states in result fst
        for state, arcs in enumerate(result_fst):
            if result_fst.is_final_state(state):
                result_fst.remove_final_weight(state)

        # determine append states and set them final
        append_states, append_state_candidates = get_append_states(result_fst, window_size, path_dict, output_dict)
        for state in append_states:
            result_fst.set_final_weight(state, 0.0)

        #print('BEFORE CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)

        # concatenate result fst and partial result
        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)

        result_fst.concatenate(partial_fst)

        #print("number of states :", result_fst.number_of_states())
        #print('PARTIAL RESULT PATHS')
        #print_output_paths(partial_fst)
        #print('AFTER CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)
        #result_fst.n_best(100)

        result_fst = hfst.HfstBasicTransducer(result_fst)

        path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(result_fst)

        # merge word borders
        # merge_word_borders(result_fst, path_dict, predecessor_dict)

        #print('AFTER MERGE RESULT PATHS')
        #print_output_paths(result_fst)

    return result_fst



def combine_results(result_list, window_size):

    #for fst in result_list:
    #    print_output_paths(fst)

    starting_fst = result_list[0].copy()
    result_fst = hfst.HfstBasicTransducer(starting_fst)
    #result_fst = remove_redundant_paths(result_fst)
    #result_fst = hfst.HfstTransducer(result_fst)
    #result_fst = hfst.HfstBasicTransducer(result_fst)

    path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(result_fst)
    merge_word_borders(result_fst, path_dict, predecessor_dict)

    for i, fst in enumerate(result_list[1:]):

        #print(i)

        partial_fst = hfst.HfstBasicTransducer(fst)
        #partial_fst = remove_redundant_paths(partial_fst)
        #partial_fst = hfst.HfstTransducer(partial_fst)
        #partial_fst = hfst.HfstBasicTransducer(partial_fst)
        #merge_word_borders(partial_fst)

        #  remove final states in result fst
        for state, arcs in enumerate(result_fst):
            if result_fst.is_final_state(state):
                result_fst.remove_final_weight(state)

        # determine append states and set them final
        append_states, append_state_candidates = get_append_states(result_fst, window_size, path_dict, output_dict)
        for state in append_states:
            result_fst.set_final_weight(state, 0.0)

        # if no word borders exist where result can be appended, append in
        # the middle of a word with a high weight
        if append_states == []:
            #print("APPEND STATES EMPTY")

            for state in append_state_candidates:
                result_fst.set_final_weight(state, 100.0)


        #print('BEFORE CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)

        # concatenate result fst and partial result
        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)

        result_fst.concatenate(partial_fst)

        #print("number of states :", result_fst.number_of_states())

        #print('PARTIAL RESULT PATHS')
        #print_output_paths(partial_fst)

        #print('AFTER CONCATENATION RESULT PATHS')
        #print_output_paths(result_fst)

        #result_fst.n_best(100)

        result_fst = hfst.HfstBasicTransducer(result_fst)


        path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(result_fst)
        merge_word_borders(result_fst, path_dict, predecessor_dict)

        #print('AFTER MERGE RESULT PATHS')
        #print_output_paths(result_fst)

        #output_dict, path_dict = merge_word_borders(result_fst)

    return result_fst


def create_result_transducer(input_str, window_size, words_per_window, error_transducer, lexicon_transducer, result_num, flag_encoder, composition=None):

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

    complete_output = combine_results(output_list, window_size)
    #complete_output = remove_redundant_paths(complete_output)

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

    print(remove_space_transducer)

    return remove_space_transducer



def get_flag_acceptor(flag_encoder):

    #flag_list = []
    #for i in range(flag_encoder.first_input_int, flag_encoder.max_input_int + 1):
    #    flag_list.append(flag_encoder.encode(i) + ':' + flag_encoder.encode(i))

    #flag_list.append(hfst.EPSILON + ':' + hfst.EPSILON)
    #print(flag_list)

    #flag_acceptor = hfst.regex('[' + ' | '.join(flag_list) + ']')

    #print(flag_acceptor)

    #return flag_acceptor

    flag_list = []
    for i in range(flag_encoder.first_input_int, flag_encoder.max_input_int + 1):
        flag_list.append(flag_encoder.encode(i))


    #flag_list.append(hfst.EPSILON)

    flag_acceptor = hfst.HfstBasicTransducer()
    tok = hfst.HfstTokenizer()
    for flag in flag_list:
        flag_acceptor.disjunct(tok.tokenize(flag), 0.0)

    print(flag_acceptor)
    flag_acceptor = hfst.HfstTransducer(flag_acceptor)
    flag_acceptor.optionalize()

    return flag_acceptor



    # Multichar

    flag_acceptor = hfst.HfstBasicTransducer()
    flag_acceptor.add_state(1)
    flag_acceptor.set_final_weight(1, 0.0)

    for symbol in flag_list:
        flag_acceptor.add_transition(0, 1, symbol, symbol, 0.0)

    flag_acceptor = hfst.HfstTransducer(flag_acceptor)
    flag_acceptor.optionalize()

    print('Flag Acceptor: ', flag_acceptor)

    return flag_acceptor


def load_transducers_bracket(error_file, punctuation_file, lexicon_file,
open_bracket_file, close_bracket_file, flag_encoder, composition_depth=1,
words_per_window=3, morphology_file=None):

    # load transducers

    flag_acceptor = get_flag_acceptor(flag_encoder)

    error_transducer = et.load_transducer(error_file)
    error_transducer.substitute((' ', hfst.EPSILON), get_edit_space_transducer(flag_encoder))

    punctuation_transducer = et.load_transducer(punctuation_file)
    punctuation_transducer.optionalize()

    open_bracket_transducer = et.load_transducer(open_bracket_file)
    open_bracket_transducer.optionalize()
    close_bracket_transducer = et.load_transducer(close_bracket_file)
    close_bracket_transducer.optionalize()

    lexicon_transducer = et.load_transducer(lexicon_file)

    space_transducer = hfst.regex('% :% ')

    # add pruned morphology to lexicon

    if morphology_file != None:
        morphology_transducer = et.load_transducer(morphology_file)
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
    result_lexicon_transducer.concatenate(flag_acceptor)
    result_lexicon_transducer.optionalize()

    result_lexicon_transducer.repeat_n(words_per_window)

    return error_transducer, result_lexicon_transducer



def load_transducers_inter_word(error_file,\
        lexicon_file,\
        punctuation_left_file,\
        punctuation_right_file,\
        flag_encoder,\
        words_per_window = 3,\
        composition_depth = 1):

    # load transducers

    error_transducer = et.load_transducer(error_file)

    punctuation_left_transducer = et.load_transducer(punctuation_left_file)
    punctuation_right_transducer = et.load_transducer(punctuation_right_file)

    punctuation_left_transducer.optionalize()
    punctuation_right_transducer.optionalize()

    lexicon_transducer = et.load_transducer(lexicon_file)

    space_transducer = hfst.regex('% :% ')

    ## add pruned morphology to lexicon

    #if morphology_file != None:
    #    morphology_transducer = et.load_transducer(morphology_file)
    #    morphology_transducer.n_best(100)
    #    lexicon_transducer.compose(morphology_transducer)

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

    # TODO: handle flag diacritics here (concatenate where?)

    #final_result_lexicon_transducer = flag_acceptor.copy()
    #final_result_lexicon_transducer.concatenate(output_lexicon)
    #final_result_lexicon_transducer.concatenate(flag_acceptor)

    return error_transducer, output_lexicon


def window_size_1(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):

    window_1 = create_result_transducer(input_str, 1, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)

    return window_1

def window_size_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):

    if ' ' not in input_str:
        return window_size_1(input_str, error_transducer, lexicon_transducer, result_num)

    window_2 = create_result_transducer(input_str, 2, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)

    return window_2


def window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num=10, composition=None):

    if ' ' not in input_str:
        return window_size_1(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num, composition)

    # create results transducers for window sizes 1 and 2, disjunct and merge states

    window_1 = create_result_transducer(input_str, 1, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)

    window_2 = create_result_transducer(input_str, 2, 3, error_transducer, lexicon_transducer, result_num, flag_encoder, composition)
    window_1.disjunct(window_2)

    complete_output_basic = hfst.HfstBasicTransducer(window_1)

    before_merge = time.time()

    path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(complete_output_basic)
    merge_word_borders(complete_output_basic, path_dict, predecessor_dict)

    after_merge = time.time()

    print('Merge Time: ', after_merge - before_merge)

    complete_output = hfst.HfstTransducer(complete_output_basic)

    return complete_output


class FlagEncoder:

    #int_char_dict
    #char_int_dict

    #first_int
    #max_int

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
    input_str = "Philoſophenvon Fndicn dur<h Gricche nland bls"
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

    #window_size = 2
    words_per_window = 3
    composition_depth = 1

    result_num = 10

    error_transducer, lexicon_transducer =\
        load_transducers_bracket(\
        'transducers/max_error_3_context_23_dta.hfst',\
        'transducers/punctuation_transducer_dta.hfst',\
        'transducers/lexicon_transducer_dta.hfst',\
        'transducers/open_bracket_transducer_dta.hfst',\
        'transducers/close_bracket_transducer_dta.hfst',\
        flag_encoder,\
        composition_depth = composition_depth,\
        words_per_window = words_per_window)
        #'transducers/morphology_with_identity.hfst')

    #error_transducer, lexicon_transducer =\
    #    load_transducers_inter_word('transducers/max_error_3_context_23_dta.hfst',\
    #    'transducers/lexicon_transducer_dta.hfst',\
    #    'transducers/left_punctuation.hfst',\
    #    'transducers/right_punctuation.hfst',\
    #    flag_encoder,\
    #    words_per_window = words_per_window,\
    #    composition_depth = composition_depth)

    #error_transducer, lexicon_transducer =\
    #    load_transducers('transducers/max_error_3_context_23_dta.hfst',\
    #    'transducers/lexicon_transducer_dta.hfst',\
    #    'transducers/any_punctuation.hfst',\
    #    'transducers/any_punctuation.hfst',\
    #    words_per_window = words_per_window,\
    #    composition_depth = composition_depth)

    openfst = True # use OpenFST for composition?

    if openfst:

        # write lexicon and error transducer in OpenFST format

        error_filename = 'error.ofst'
        lexicon_filename = 'lexicon.ofst'
        error_filename_b = b'error.ofst'
        lexicon_filename_b = b'lexicon.ofst'

        for filename, fst in [(error_filename, error_transducer), (lexicon_filename, lexicon_transducer)]:
            out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
            out.write(fst)
            out.flush()
            out.close()

        # generate Composition Object

        composition = pyComposition(error_filename_b, lexicon_filename_b, result_num)
        print(composition)

        preparation_done = time.time()
        print('Preparation Time: ', preparation_done - start)

        # apply correction using Composition Object

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num, composition)


    else:

        preparation_done = time.time()
        print('Preparation Time: ', preparation_done - start)

        # apply correction directly in hfst

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num)

    # write output to filesystem

    out = hfst.HfstOutputStream(filename='output/' + input_str + '.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(complete_output)
    out.flush()
    out.close()


    print('Complete Output')
    print_output_paths(complete_output)

    complete_output.n_best(100)

    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=100, max_cycles=0)

    output_list = []

    for input, outputs in complete_paths.items():
        for output in outputs:
            text = output[0].replace('@_EPSILON_SYMBOL_@', '')
            if not text in output_list:
                output_list.append(text)

    print('Output List')
    for item in output_list:
        print(item)

    ## load and apply language model

    #lm_file = 'lang_mod_theta_0_000001.mod.modified.hfst'
    #lowercase_file = 'lowercase.hfst'

    #lm_fst = et.load_transducer(lm_file)
    #lowercase_fst = et.load_transducer(lowercase_file)

    #complete_output.output_project()

    #complete_output.compose(lowercase_fst)

    #print('Lowercase Output')
    #print_output_paths(complete_output)

    #complete_output.compose(lm_fst)

    #print('Language Model Output')
    #print_output_paths(complete_output)

    ## get 10 paths

    ##complete_output = complete_output.n_best(10)
    ##complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=10, max_cycles=0)

    ##for input, outputs in complete_paths.items():
    ##    #print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
    ##    for output in outputs:
    ##        print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))


if __name__ == '__main__':
    main()
