import math
import hfst
import libhfst
import error_transducer as et


def merge_word_borders(basic_fst):

    initial_state = 0

    path_dict = {}          # input string leading to state
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
        else:
            path_dict[actual_state] = ''

        # add new states to queue
        transitions = basic_fst.transitions(actual_state)
        for transition in transitions:
            target = transition.get_target_state()
            if target not in visited and target not in queue:
                predecessor_dict[target] = actual_state
                transition_dict[target] = transition
                queue.append(target)

   # find states with space as input symbol
    input_space_states = []
    for state, transitions in enumerate(basic_fst):
        for transition in transitions:
            if transition.get_output_symbol() == ' ':
                input_space_states.append(transition.get_target_state())

    # states should merge before the space
    merge_candidates = []
    for state in input_space_states:
        merge_candidates.append(predecessor_dict[state])

    # TODO: müsste man hier nicht mit der gesamten Menge von states
    # vergleichen statt nur mit denen vor einem Leerzeichen? der neueste
    # letzte Zustand hat kein Leerzeichen am Ende, sondern ist am Wortende

    # find pairs of corresponding states and merge
    for i, state1 in enumerate(merge_candidates):
        for state2 in merge_candidates[i:]:
            if path_dict[state1] == path_dict[state2]:
                transition_exists = False

                for transition in basic_fst.transitions(state1):
                    if transition.get_target_state() == state2 \
                    and transition.get_input_symbol() == hfst.EPSILON \
                    and transition.get_output_symbol() == hfst.EPSILON:
                        transition_exists = True

                if not transition_exists:
                    basic_fst.add_transition(state1, state2, hfst.EPSILON, hfst.EPSILON, 0.0)
                    basic_fst.add_transition(state2, state1, hfst.EPSILON, hfst.EPSILON, 0.0)

    return path_dict


def create_input_transducer(input_str):

    fst_dict = {input_str: [(input_str, -math.log(1))]}
    input_fst = hfst.fst(fst_dict)
    return input_fst


def prepare_input(input_str, window_size):

    splitted = input_str.split(' ')
    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]
        input_list.append(' '.join(single_input) + ' ')

    return input_list


def compose_and_search(input_str, error_transducer, lexicon_transducer, result_num):

    input_fst = create_input_transducer(input_str)

    input_fst.compose(error_transducer)
    input_fst.compose(lexicon_transducer)

    result_num = 10

    input_fst.n_best(result_num)
    #results = input_fst.extract_paths(max_cycles=0, max_number=5, output='dict')

    #results = input_fst.extract_paths(max_number=result_num)

    input_fst.n_best(result_num)

    results = input_fst

    return results


#def prepare_single_result(single_result):
#
#    for state in single_result:



def combine_results(result_list):

    starting_fst = result_list[0]

    starting_fst.push_weights_to_start()
    result_fst = hfst.HfstBasicTransducer(starting_fst)
    path_dict = merge_word_borders(result_fst)

    final_states = {}

    starting_final_states = []
    for state in result_fst.states():
        input_string = path_dict[state]
        if len(input_string) > 0:
            if input_string[-1] == ' ' and input_string.count(' ') == 1:
                starting_final_states.append(state)

    # find states with input space
    final_states[-1] = starting_final_states

    for i, fst in enumerate(result_list[1:]):

        print(i)

        fst.prune()
        fst.push_weights_to_start()
        partial_fst = hfst.HfstBasicTransducer(fst)

        merge_word_borders(partial_fst)

        last_final_list = []

        input_space_states = []

        #for state in result_fst.states():
        #    if result_fst.is_final_state(state):
        #        last_final_list.append(state)
        #        result_fst.remove_final_weight(state)

        for state, arcs in enumerate(result_fst):
            for arc in arcs:
                if arc.get_output_symbol() == ' ':
                    input_space_states.append(arc.get_target_state())
            if result_fst.is_final_state(state):
                last_final_list.append(state)
                result_fst.remove_final_weight(state)
        final_states[i] = last_final_list

        print("Input Space States")
        print(input_space_states)

        #for state in input_space_states:
        #    if i >= 2:
        #        if state in final_states[i-2]:
        #            result_fst.set_final_weight(state, 0.0)
        #    else:
        #        result_fst.set_final_weight(state, 0.0)

        for state in final_states[i-1]:
            result_fst.set_final_weight(state, 0.0)

        #final_states.append(last_final_list)

        result_fst = hfst.HfstTransducer(result_fst)
        partial_fst = hfst.HfstTransducer(partial_fst)
        result_fst.concatenate(partial_fst)
        result_fst = hfst.HfstBasicTransducer(result_fst)
        merge_word_borders(result_fst)

        #for state in partial_fst.states():





    return result_fst


def main():

    #input_str = "bIeibt"
    #input_str = "{CAP}unterlagen"
    #input_str = "fur"
    #input_str = "das miüssenwirklich seHr sclnöne bla ue sein"
    input_str = "das miüssenwirklich seHr sclnöne bla ue sei n"

    window_size = 2
    result_num = 10

    # create transducers

    error_transducer = et.load_transducer('transducers/max_error_3.hfst')

    lexicon_transducer = et.load_transducer('transducers/lexicon.hfst')
    space_transducer = hfst.regex('% :% ')
    lexicon_transducer.concatenate(space_transducer)
    lexicon_transducer.optionalize()
    lexicon_transducer.repeat_n(3)

    input_list = prepare_input(input_str, window_size)
    output_list = []

    for single_input in input_list:

        print('\n')

        results = compose_and_search(single_input, error_transducer, lexicon_transducer, result_num)
        output_list.append(results)


        output_dict = results.extract_paths(max_number=result_num)

        for input, outputs in output_dict.items():
            print('%s:' % input)
            for output in outputs:
                print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))

    print('\n')

    complete_output = combine_results(output_list)
    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=result_num)

    for input, outputs in complete_paths.items():
        print('%s:' % input)
        for output in outputs:
            print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))


if __name__ == '__main__':
    main()
