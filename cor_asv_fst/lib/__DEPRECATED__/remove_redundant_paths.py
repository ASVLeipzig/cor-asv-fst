import hfst


def remove_redundant_paths(basic_fst):

    initial_state = 0

    # key: state_id, value: tuple containing three dicts
    # predecessor -> output_string, output_string -> predecessor,
    # output_string -> weight, string -> transition
    path_dict = {}

    states = basic_fst.states()
    num_states = len(states)
    visited = []
    queue = [initial_state]

    #path_dict[initial_state] = [('', 0.0, None)]
    path_dict[initial_state] = ({-1: ['']}, {'': -1}, {'': 0.0}, {'': None}, {None: ['']})


    #while len(visited) < num_states:
    while queue != []:

        actual_state = queue.pop()
        visited.append(actual_state)

        actual_pre_strings_dict = path_dict[actual_state][0]
        actual_string_pre_dict = path_dict[actual_state][1]
        actual_string_weight_dict = path_dict[actual_state][2]
        actual_string_transition_dict = path_dict[actual_state][3]
        actual_transition_strings_dict = path_dict[actual_state][4]

        print()
        print('Actual State: ', actual_state)
        print('Queue: ', queue)
        print('Visited: ', visited)

        # iterate over outgoing transitions
        transitions = basic_fst.transitions(actual_state)
        for transition in transitions:

            target = transition.get_target_state()
            print()
            print('Target: ', target)

            transition_string = str(actual_state) + ' ' + transition.__str__()
            print('Transition: ', transition_string)

            strings_improved = []
            strings_worsened = []
            strings_equal = []


            target_pre_strings_dict = {}
            target_string_pre_dict = {}
            target_string_weight_dict = {}
            target_string_transition_dict = {}
            target_transition_strings_dict = {}

            print('Path Dict Keys: ', path_dict.keys())
            target_dicts = path_dict.get(target)

            if target_dicts != None:

                print('Target Dict already set.')

                target_pre_strings_dict = target_dicts[0]
                target_string_pre_dict = target_dicts[1]
                target_string_weight_dict = target_dicts[2]
                target_string_transition_dict = target_dicts[3]
                target_transition_strings_dict = target_dicts[4]


            for output_string in list(actual_string_pre_dict.keys()):

                print('String at predecessor: ', output_string)

                new_output_symbol = transition.get_output_symbol()
                if new_output_symbol == hfst.EPSILON:
                    new_output_symbol = ''
                new_output_string = output_string + new_output_symbol
                print('New String: ', new_output_string)
                new_weight = actual_string_weight_dict[output_string] + transition.get_weight()

                old_weight = target_string_weight_dict.get(new_output_string)
                if old_weight != None:
                    if old_weight < new_weight:
                        strings_worsened.append(new_output_string)
                    elif old_weight == new_weight:
                        strings_equal.append(new_output_string)
                    else:
                        strings_improved.append(new_output_string)

                else:
                    strings_improved.append(new_output_string)

            # if the new transition creates no new or better
            # output_strings, remove transition
            if strings_improved == []:
                old_transitions = list(target_string_transition_dict.values())

                splitted_transition = transition_string.split()
                splitted_old_transitions = list(map(lambda x: x.split(), old_transitions))

                equivalent_transition = [x for x in splitted_old_transitions if\
                    x[0] == splitted_transition[0] and\
                    x[1] == splitted_transition[1] and\
                    x[2] == splitted_transition[2] and\
                    x[3] == splitted_transition[3]]

                if transition_string not in old_transitions:
                    basic_fst.remove_transition(actual_state, transition)
                    print('Transition led to no improvement! Remove new transition: ',\
                        actual_state, transition)
                    print('Old transitions: ', list(target_string_transition_dict.values()))

                    # add equivalent transition (same starting state),
                    # because it gets removed with remove_transition
                    # (ignored different weights)
                    if equivalent_transition != []:
                        good_old_transition = equivalent_transition[0]
                        #splitted = good_old_transition.split()
                        splitted = good_old_transition
                        basic_fst.add_transition(int(splitted[0]),\
                            hfst.HfstBasicTransition(int(splitted[1]), splitted[2], splitted[3], float(splitted[4])))

            else:

                print('Improved Strings: ', strings_improved)
                print('Worsened Strings: ', strings_worsened)
                print('Equal Strings: ', strings_equal)


                removal_candidates = []
                new_candidates = []

                for string in strings_improved:
                    pred = target_string_pre_dict.get(string)
                    if pred != None:
                        removal_candidates.append(string)
                    else:
                        new_candidates.append(string)
                for string in strings_equal:
                    pred = target_string_pre_dict.get(string)
                    if pred != None:
                        removal_candidates.append(string)
                    else:
                        new_candidates.append(string)


                print('Removal Candidates: ', removal_candidates)

                removal_transitions = []

                for removal_candidate in removal_candidates:
                    old_predecessor = target_string_pre_dict.get(removal_candidate)
                    old_transition = target_string_transition_dict.get(removal_candidate)
                    old_transition_strings = target_transition_strings_dict.setdefault(removal_candidate, [])

                    remove = True

                    for tr in old_transition_strings:
                        if tr not in removal_candidates:
                            remove = False

                    if remove:
                        removal_transitions.append((old_predecessor, old_transition))


                # remove redundant transitions

                for st, tr in removal_transitions:
                    if tr != transition_string:
                        splitted = transition_string.split()
                        basic_fst.remove_transition(\
                            st, hfst.HfstBasicTransition(\
                            int(splitted[1]), splitted[2], splitted[3], float(splitted[4])))
                        print('Remove old redundant transition: ', st, tr)
                        print('New better transition: ', print(transition_string))

                        # add new better transition, if it has same
                        # starting state, because remove_transition removes
                        # it ignoring different weight
                        if st == int(splitted[0]):
                            basic_fst.add_transition(actual_state, transition)

                # update dicts

                #all_candidates = removal_candidates + new_candidates
                all_candidates = strings_improved + strings_equal
                print('All candidates: ', all_candidates)

                target_pre_strings_dict[actual_state] = all_candidates

                for string in all_candidates:
                    target_string_pre_dict[string] = actual_state

                    new_weight = actual_string_weight_dict[string[0:-1]] + transition.get_weight()
                    print('Predecessor weight: ', actual_string_weight_dict[string[0:-1]])
                    print('Target weight: ', new_weight)
                    target_string_weight_dict[string] = new_weight

                    target_string_transition_dict[string] = transition_string

                target_transition_strings_dict[transition_string] = all_candidates


                new_target_dicts =\
                    (target_pre_strings_dict, target_string_pre_dict,\
                    target_string_weight_dict, target_string_transition_dict,\
                    target_transition_strings_dict)

                path_dict[target] = new_target_dicts
                print('String-Weight Keys: ', path_dict[target][2].keys())

                # add target to queue

                if target not in queue:
                    queue.append(target)

    return basic_fst




def main():

    t = hfst.HfstBasicTransducer()
    t.add_state(1)
    t.add_state(2)
    t.add_state(3)
    t.add_state(4)
    t.add_state(5)
    t.add_state(6)
    t.add_state(7)
    t.add_transition(0, 1, 'a', 'a', 1.0)
    t.add_transition(0, 2, 'a', 'a', 2.0)
    t.add_transition(0, 5, 'a', 'b', 1.0)
    t.add_transition(0, 5, 'a', 'b', 2.0)
    t.add_transition(1, 3, 'a', 'a', 1.0)
    t.add_transition(2, 3, 'a', 'a', 1.0)
    t.add_transition(3, 4, 'a', 'b', 1.0)
    t.add_transition(4, 7, 'a', 'd', 1.0)
    t.add_transition(5, 6, 'a', 'c', 1.0)
    t.add_transition(6, 7, 'a', 'd', 1.0)
    t.set_final_weight(7, 0.0)


    t = remove_redundant_paths(t)

    print('Output FST')
    print(t)

    non_basic_fst = hfst.HfstTransducer(t)

    try:
    # Extract paths and remove tokenization
        results = non_basic_fst.extract_paths(output='dict')
    except hfst.exceptions.TransducerIsCyclicException as e:
    # This should not happen because transducer is not cyclic.
        print("TEST FAILED")
        exit(1)

    for input, outputs in results.items():
        print('%s:' % input)
        for output in outputs:
            print('  %s\t%f' % (output[0], output[1]))


    return


if __name__ == '__main__':
    main()
