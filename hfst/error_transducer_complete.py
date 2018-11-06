import argparse
import hfst

import helper


def main():
    """
    Take the simple error transducers for different context sizes,
    and combine them to full, mixed error transducers, which can 
    correct multiple errors (given in max_errors) and 
    within different extents if context (given in max_context).
    """
    
    parser = argparse.ArgumentParser(description='OCR post-correction ocrd-cor-asv-fst error model creator')
    parser.add_argument('-E', '--max-errors', metavar='NUM', type=int, default=3, help='maximum number of errors the resulting FST can correct (applicable within one window, i.e. a certain number of words)')
    parser.add_argument('-C', '--max-context', metavar='NUM', type=int, default=3, help='maximum size of context to generate FSTs for (mixed by disjunction without back-off)')
    args = parser.parse_args()
    
    error_transducers = {}
    for n in range(1,args.max_errors+1):
        error_transducers[n] = helper.load_transducer('error_transducer_' + str(n) + '.hfst')

    contexts = []
    for n in range(1,args.max_context+1):
        for m in range(1,n+1):
            contexts.append(list(range(m,n+1)))
    
    #error_transducer_1 = helper.load_transducer('error_transducer_1.hfst')
    #error_transducer_2 = helper.load_transducer('error_transducer_2.hfst')
    #error_transducer_3 = helper.load_transducer('error_transducer_3.hfst')
    
    # FIXME: make proper back-off transducer
    
    #error_numbers = [1, 2, 3, 4, 5]
    #error_numbers = [1, 2, 3]
    #contexts = ['1', '2', '3', '123', '23'] # all possible contexts

    acceptor = hfst.regex('?')
    acceptor.repeat_star()
    acceptor.minimize()

    for context in contexts:

        print('Context: ', context)

        error_transducer = hfst.HfstTransducer()

        for n in range(1,args.max_errors+1):
            if n in context:
                error_transducer.disjunct(error_transducers[n])

        one_error = acceptor.copy()
        one_error.concatenate(error_transducer)
        one_error.optionalize()

        #error_transducer = acceptor.copy()
        for num_errors in range(1,args.max_errors+1):
            print('Number of errors:', num_errors)
            
            result_transducer = one_error.copy()
            result_transducer.repeat_n(num_errors)
            #result_transducer.disjunct(error_transducer) # N ∪ N-1 # deteriorates model
            
            error_transducer =  result_transducer
            error_transducer.concatenate(acceptor)

            helper.save_transducer('max_error_' + str(num_errors) + '_context_'+ ''.join(map(str,context)) + '.hfst', error_transducer)


if __name__ == '__main__':
    main()
