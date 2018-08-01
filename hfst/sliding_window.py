import math
import hfst
import libhfst
import error_transducer as et

import time

from composition import pyComposition


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


def prepare_input(input_str, window_size):
    """Takes long input_string and splits it at space characters into a list of windows with
    window_size words."""

    splitted = input_str.split(' ')

    #if len(splitted) < window_size:
    #    return ' '.join(splitted) + ' '

    input_list = []

    for i in range(0, len(splitted) - window_size + 1):
        single_input = splitted[i:i + window_size]
        input_list.append(' '.join(single_input) + ' ')

    return input_list


def compose_and_search(input_str, error_transducer, lexicon_transducer, result_num, composition = None):

    input_fst = create_input_transducer(input_str)

    if composition != None:

        composition.compose(input_str.encode())

        result_fst = et.load_transducer('output/' + input_str + '.fst')
        #result_fst = et.load_transducer(input_str + '.fst')

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


def combine_results(result_list, window_size):

    #for fst in result_list:
    #    print_output_paths(fst)

    starting_fst = result_list[0].copy()
    result_fst = hfst.HfstBasicTransducer(starting_fst)

    path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(result_fst)
    merge_word_borders(result_fst, path_dict, predecessor_dict)

    for i, fst in enumerate(result_list[1:]):

        #print(i)

        partial_fst = hfst.HfstBasicTransducer(fst)
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


def create_result_transducer(input_str, window_size, words_per_window, error_transducer, lexicon_transducer, result_num, composition=None):

    #lexicon_transducer.repeat_n(words_per_window)

    start = time.time()

    input_list = prepare_input(input_str, window_size)
    #print("Input List: ", input_list)
    output_list = []

    for i, single_input in enumerate(input_list):
        #print("Single Input: ", single_input)

        results = compose_and_search(single_input, error_transducer, lexicon_transducer, result_num, composition)

        output_list.append(results)

    after_composition = time.time()

    complete_output = hfst.HfstTransducer(combine_results(output_list, window_size))

    after_combination = time.time()

    print('Composition Time: ', after_composition - start)
    print('Combination Time: ', after_combination - after_composition)

    return complete_output


def load_transducers_bracket(error_file, punctuation_file, lexicon_file,
open_bracket_file, close_bracket_file, composition_depth=1,
words_per_window=3, morphology_file=None):

    # load transducers

    error_transducer = et.load_transducer(error_file)

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

    result_lexicon_transducer = open_bracket_transducer.copy()
    result_lexicon_transducer.concatenate(lexicon_transducer)
    result_lexicon_transducer.concatenate(punctuation_transducer)
    result_lexicon_transducer.concatenate(close_bracket_transducer)
    result_lexicon_transducer.concatenate(space_transducer)
    result_lexicon_transducer.optionalize()

    result_lexicon_transducer.repeat_n(words_per_window)

    return error_transducer, result_lexicon_transducer



def load_transducers_inter_word(error_file,\
        lexicon_file,\
        punctuation_left_file,\
        punctuation_right_file,\
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

    return error_transducer, output_lexicon




def window_size_1(input_str, error_transducer, lexicon_transducer, result_num=10, composition=None):

    window_1 = create_result_transducer(input_str, 1, 3, error_transducer, lexicon_transducer, result_num, composition)

    return window_1

def window_size_2(input_str, error_transducer, lexicon_transducer, result_num=10, composition=None):

    if ' ' not in input_str:
        return window_size_1(input_str, error_transducer, lexicon_transducer, result_num)

    window_2 = create_result_transducer(input_str, 2, 3, error_transducer, lexicon_transducer, result_num, composition)

    return window_2


def window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num=10, composition=None):

    if ' ' not in input_str:
        return window_size_1(input_str, error_transducer, lexicon_transducer, result_num, composition)

    # create results transducers for window sizes 1 and 2, disjunct and merge states

    window_1 = create_result_transducer(input_str, 1, 3, error_transducer, lexicon_transducer, result_num, composition)

    window_2 = create_result_transducer(input_str, 2, 3, error_transducer, lexicon_transducer, result_num, composition)
    window_1.disjunct(window_2)

    complete_output_basic = hfst.HfstBasicTransducer(window_1)

    before_merge = time.time()

    path_dict, output_dict, transition_dict, predecessor_dict = get_path_dicts(complete_output_basic)
    merge_word_borders(complete_output_basic, path_dict, predecessor_dict)

    after_merge = time.time()

    print('Merge Time: ', after_merge - before_merge)

    complete_output = hfst.HfstTransducer(complete_output_basic)

    return complete_output


def main():

    start = time.time()

    #input_str = "bIeibt"
    #input_str = "{CAP}unterlagen"
    #input_str = "fur"
    #input_str = "das miüssenwirklich seHr sclnöne bla ue Schuhe seln"
    #input_str = "Das miüssenwirklich seHr sclnöne bla ue sein"
    #input_str = "miü ssen miü ssen wirklich"# seHr sclnöne bla ue Schuhe seln"
    #input_str = "sclnöne bla ue sein"
    #input_str = "Philoſophen von Fndicn dur<h Gricchenland bis"
    #input_str = "wirrt ſah fie um ſh, und als fie den Mann mit dem"

    #window_size = 2
    words_per_window = 3
    composition_depth = 1

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

    #input_str = 'er das Fräulein auf einem ſchönen Zelter unten rider'
    #input_str = 'er das Fräulein auf einem ſchönen Zelter unten rider'

    #input_str.strip('\n\u000C')

    result_num = 10

    error_transducer, lexicon_transducer =\
        load_transducers_bracket(\
        'transducers/max_error_3_context_23_dta.hfst',\
        'transducers/punctuation_transducer_dta.hfst',\
        'transducers/lexicon_transducer_dta.hfst',\
        'transducers/open_bracket_transducer_dta.hfst',\
        'transducers/close_bracket_transducer_dta.hfst',\
        composition_depth = composition_depth,\
        words_per_window = words_per_window)
        #'transducers/morphology_with_identity.hfst')

    #lexicon_transducer.repeat_n(words_per_window)

    #error_transducer, lexicon_transducer =\
    #    load_transducers_inter_word('transducers/max_error_3_context_23_dta.hfst',\
    #    'transducers/lexicon_transducer_dta.hfst',\
    #    'transducers/left_punctuation.hfst',\
    #    'transducers/right_punctuation.hfst',\
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

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num, composition)


    else:

        preparation_done = time.time()
        print('Preparation Time: ', preparation_done - start)

        # apply correction directly in hfst

        complete_output = window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num)



    #out = hfst.HfstOutputStream(filename='output/' + input_str + '.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    #out.write(complete_output)
    #out.flush()
    #out.close()


    # load and apply language model

    print('Complete Output')
    print_output_paths(complete_output)


    complete_output.n_best(100)

    ##complete_output.determinize()
    ##complete_output.minimize()



    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=10000, max_cycles=0)

    #output_list = []

    #for input, outputs in complete_paths.items():
    #    for output in outputs:
    #        text = output[0].replace('@_EPSILON_SYMBOL_@', '')
    #        if not text in output_list:
    #            output_list.append(text)

    #print('Output List')
    #for item in output_list:
    #    print(item)





    out = hfst.HfstOutputStream(filename='output/' + input_str + '.hfst', hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
    out.write(complete_output)
    out.flush()
    out.close()



    lm_file = 'lang_mod_theta_0_000001.mod.modified.hfst'
    lowercase_file = 'lowercase.hfst'

    lm_fst = et.load_transducer(lm_file)
    lowercase_fst = et.load_transducer(lowercase_file)

    complete_output.output_project()

    complete_output.compose(lowercase_fst)

    print('Lowercase Output')
    print_output_paths(complete_output)

    complete_output.compose(lm_fst)

    print('Language Model Output')
    print_output_paths(complete_output)





    # get 10 paths

    #complete_output = complete_output.n_best(10)
    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=10, max_cycles=0)

    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1000, max_cycles=0)
    #print(list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', ''))


    #complete_output.n_best(result_num)
    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=result_num, max_cycles=0)

    #complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=10000, max_cycles=0)

    #output_list = []

    #for input, outputs in complete_paths.items():
    #    for output in outputs:
    #        text = output[0].replace('@_EPSILON_SYMBOL_@', '')
    #        if not text in output_list:
    #            output_list.append(text)

    #print('Output List')
    #for item in output_list:
    #    print(item)

    #complete_output.n_best(result_num)
    #print_output_paths(complete_output)

    #complete_paths = hfst.HfstTransducer(complete_output).extract_shortest_paths()

    #for input, outputs in complete_paths.items():
    #    #print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
    #    for output in outputs:
    #        print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))


if __name__ == '__main__':
    main()
