
import math
import hfst
import libhfst
import error_transducer as et


def create_input_transducer(input_str):

    fst_dict = {input_str: [(input_str, -math.log(1))]}

    input_fst = hfst.fst(fst_dict)

    return input_fst


def main():

    #input_str = "bIeibt"
    #input_str = "{CAP}unterlagen"
    #input_str = "fur"

    error_transducer_1 = et.load_transducer('error_transducer_1.hfst')
    error_transducer_2 = et.load_transducer('error_transducer_2.hfst')
    error_transducer_3 = et.load_transducer('error_transducer_3.hfst')

    # allow one error

    acceptor = hfst.regex('?')
    acceptor.repeat_star()
    acceptor.minimize()

    #print(acceptor)

    #acceptor.remove_epsilons()
    #acceptor.minimize()

    error_transducer = error_transducer_1.copy()
    error_transducer.disjunct(error_transducer_2)
    error_transducer.disjunct(error_transducer_3)

    #print(error_transducer)

    one_error = acceptor.copy()
    one_error.concatenate(error_transducer)
    #one_error.concatenate(acceptor)
    one_error.optionalize()

    #print(one_error)

    error_number = 3
    print('Number of errors:', error_number)

    one_error.repeat_n(error_number)

    #one_error.minimize()

    ## allow two errors

    #two_errors = one_error.copy()
    #two_errors.concatenate(error_transducer)
    #two_errors.concatenate(acceptor)

    ## allow three errors

    #three_errors = two_errors.copy()
    #three_errors.concatenate(error_transducer)
    #three_errors.concatenate(acceptor)

    error_transducer = one_error
    error_transducer.concatenate(acceptor)

    et.save_transducer('max_error_' + str(error_number) + '.htsf', error_transducer)

    # compose input, error model, and lexicon

    input_fst = create_input_transducer(input_str)

    input_fst.compose(error_transducer)
    #input_fst.output_project()
    #input_fst.remove_epsilons()

    #lexicon_transducer = et.load_transducer('fst/lexicon.fsm')
    #morphology_transducer = et.load_transducer('fst/rules.fsm')

    lexicon_transducer = et.load_transducer('wwm/extended_lexicon_inverted_projected.tropical')

    input_fst.compose(lexicon_transducer)

    result_num = 100

    input_fst.n_best(result_num)
    #results = input_fst.extract_paths(max_cycles=0, max_number=5, output='dict')

    results = input_fst.extract_paths(max_number=result_num)

    #except libhfst.TransducerIsCyclicException as e:

    #    print('failed')
    #    exit(1)

    for input, outputs in results.items():
        print('%s:' % input)
        for output in outputs:
            print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))



    #results = input_fst.extract_paths(output='dict')

    #for input, outputs in results.items():
    #    print('%s:' % input)
    #    for output in outputs:
    #        print(' %s\t%f' % (output[0], output[1]))


if __name__ == '__main__':
    main()
