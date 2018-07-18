import hfst
import lexicon_transducer as lt
import error_transducer as et

def main():

    symbols_per_side = 4

    punctuation_file = 'dta_punctuation_any.txt'
    output_hfst = 'any_punctuation.hfst'

    scaling_transducer = hfst.regex('?::1.0')

    print('Read', punctuation_file, '...')
    freq_list = lt.read_lexicon(punctuation_file)
    print("Construct Transducer...")
    transducer = lt.transducer_from_list(freq_list)
    transducer = hfst.HfstTransducer(transducer)

    transducer.compose(scaling_transducer)

    transducer.optionalize()

    result_transducer = transducer.copy()

    for i in range(1, symbols_per_side):
        print(i)
        next_symbol = transducer.copy()
        for j in range(1, i):
            next_symbol.compose(scaling_transducer)
        result_transducer.concatenate(next_symbol)




    et.save_transducer(output_hfst, result_transducer)


if __name__ == '__main__':
    main()
