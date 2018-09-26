import helper
import hfst


def main():
    """Takes the simple error_transducers and combines them to full-fledged
    complete error_transducers, which can correct multiple errors
    (number of error to correct specified in error_number) and different
    extents if context (defined in contexts)."""

    error_transducer_1 = helper.load_transducer('error_transducer_1.hfst')
    error_transducer_2 = helper.load_transducer('error_transducer_2.hfst')
    error_transducer_3 = helper.load_transducer('error_transducer_3.hfst')

    #error_numbers = [1, 2, 3, 4, 5]
    error_numbers = [1, 2, 3]
    contexts = ['1', '2', '3', '123', '23'] # all possible contexts

    acceptor = hfst.regex('?')
    acceptor.repeat_star()
    acceptor.minimize()

    for context in contexts:

        print('Context: ', context)

        error_transducer = hfst.HfstTransducer()

        if '1' in context:
            error_transducer.disjunct(error_transducer_1)
        if '2' in context:
            error_transducer.disjunct(error_transducer_2)
        if '3' in context:
            error_transducer.disjunct(error_transducer_3)

        one_error = acceptor.copy()
        one_error.concatenate(error_transducer)
        one_error.optionalize()

        for error_number in error_numbers:
            print('Number of errors:', error_number)

            result_transducer = one_error.copy()

            result_transducer.repeat_n(error_number)

            error_transducer = result_transducer
            error_transducer.concatenate(acceptor)

            helper.save_transducer('max_error_' + str(error_number) + '_context_'+ str(context) + '.hfst', error_transducer)


if __name__ == '__main__':
    main()
