import math
import hfst
import libhfst
import error_transducer as et


def merge_word_borders(basic_fst):

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
    ## compare candidates with all other states, because new last states do
    ## not end with a space character, but with a word
    # compare with other merge_candidates, because new last states are
    # handled by adding states without outgoing transitions
    for i, state1 in enumerate(merge_candidates):
        for state2 in merge_candidates[i:]:
        #for state2 in basic_fst.states():
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

    return output_dict, path_dict


def create_input_transducer(input_str):

    fst_dict = {input_str: [(input_str, -math.log(1))]}
    input_fst = hfst.fst(fst_dict)
    return input_fst


def prepare_input(input_str, window_size):
    """Takes long input_string and splits it at space characters into a list of windows with
    window_size words."""

    splitted = input_str.split(' ')
    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]
        input_list.append(' '.join(single_input) + ' ')

    return input_list


def compose_and_search(input_str, error_transducer, lexicon_transducer, result_num):

    input_fst = create_input_transducer(input_str)

    result_fst = input_fst.copy()

    result_fst.compose(error_transducer)
    result_fst.compose(lexicon_transducer)

    #result_fst.n_best(result_num)
    #results = input_fst.extract_paths(max_cycles=0, max_number=5, output='dict')

    #results = input_fst.extract_paths(max_number=result_num)

    result_fst.n_best(result_num)

    if result_fst.number_of_states() == 0:
        print("result empty")
        #input_fst.set_final_weights(100.0)
        input_fst = set_transition_weights(input_fst)
        return input_fst

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

    print('APPEND STRING: ', append_string)

    # take all states with the correct input string
    items = list(path_dict.items())
    filtered_items = list(filter(lambda x: x[1] == append_string, items))

    append_state_candidates = list(map(lambda x: x[0], filtered_items))
    print('APPEND STATE CANDIDATES: ', append_state_candidates)

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

    append_states = list(filter(lambda x: output_dict[x][-1] == ' ', append_state_candidates))

    print('APPEND STATES: ', append_states)

    return append_states


def print_output_paths(basic_fst):
    #complete_paths = hfst.HfstTransducer(basic_fst).extract_paths(max_number=5, max_cycles=0)
    complete_paths = hfst.HfstTransducer(basic_fst).extract_shortest_paths()
    for input, outputs in complete_paths.items():
        print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        for output in outputs:
            print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))
    print('\n')


def combine_results(result_list, window_size):

    starting_fst = result_list[0].copy()
    result_fst = hfst.HfstBasicTransducer(starting_fst)
    output_dict, path_dict = merge_word_borders(result_fst)

    for i, fst in enumerate(result_list[1:]):

        print(i)

        partial_fst = hfst.HfstBasicTransducer(fst)
        merge_word_borders(partial_fst)

        #  remove final states in result fst
        for state, arcs in enumerate(result_fst):
            if result_fst.is_final_state(state):
                result_fst.remove_final_weight(state)

        # determine append states and set them final
        append_states = get_append_states(result_fst, window_size, path_dict, output_dict)
        for state in append_states:
            result_fst.set_final_weight(state, 0.0)


        print('BEFORE CONCATENATION RESULT PATHS')
        print_output_paths(result_fst)

        # concatenate result fst and partial result
        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)

        result_fst.concatenate(partial_fst)

        print("number of states :", result_fst.number_of_states())

        print('PARTIAL RESULT PATHS')
        print_output_paths(partial_fst)

        print('AFTER CONCATENATION RESULT PATHS')
        print_output_paths(result_fst)

        #result_fst.n_best(100)

        result_fst = hfst.HfstBasicTransducer(result_fst)
        output_dict, path_dict = merge_word_borders(result_fst)

    return result_fst


def main():

    #input_str = "bIeibt"
    #input_str = "{CAP}unterlagen"
    #input_str = "fur"
    #input_str = "das miüssenwirklich seHr sclnöne bla ue sein"
    input_str = "Das miüssenwirklich seHr sclnöne bla ue sein"
    #input_str = "sclnöne bla ue sein"

    window_size = 2
    result_num = 10
    words_per_window = 3

    # create transducers

    error_transducer = et.load_transducer('transducers/max_error_3.hfst')

    lexicon_transducer = et.load_transducer('transducers/lexicon.hfst')
    #lexicon_transducer = et.load_transducer('transducers/lexicon_transducer_asse.hfst')
    space_transducer = hfst.regex('% :% ')
    lexicon_transducer.concatenate(space_transducer)
    lexicon_transducer.optionalize()
    lexicon_transducer.repeat_n(words_per_window)

    input_list = prepare_input(input_str, window_size)
    output_list = []

    for single_input in input_list:

        print('\n')

        results = compose_and_search(single_input, error_transducer, lexicon_transducer, result_num)
        output_list.append(results)

        print_output_paths(results)

        #output_dict = results.extract_paths(max_number=result_num)

        #for input, outputs in output_dict.items():
        #    print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
        #    for output in outputs:
        #        print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))

    print('\n')

    complete_output = hfst.HfstTransducer(combine_results(output_list, window_size))
    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=result_num, max_cycles=0)

    #complete_output.n_best(result_num)
    complete_output.n_best(200)
    #print_output_paths(complete_output)

    #complete_paths = hfst.HfstTransducer(complete_output).extract_shortest_paths()

    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=result_num, max_cycles=0)


    for input, outputs in complete_paths.items():
        print('%s:' % input)
        for output in outputs:
            print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))


if __name__ == '__main__':
    main()
