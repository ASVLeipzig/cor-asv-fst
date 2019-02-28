import hfst
import helper


def punctuation_characters_from_files(file_list):
    """Read a file_list of files consisting of tab-separted pairs of word/symbol
    and weight. Construct a set of characters in these files."""

    chars = set()

    for f in file_list:

        print('Read', f, '...')

        freq_list = helper.read_lexicon(f)
        for (string1, string2, weight) in freq_list:
            for char in string1:
                if char != ' ' and char != '\t':
                    chars.add(char)

    #chars.add(' ')
    return chars


def main():
    """Construct a transducer accepting up to n consecutive characters out
    of a given file_list (containing tab-separated word-weight pairs) with
    weight 0."""

    file_list = [\
        "./Fertig 6 DTA/close_bracket.txt",\
        "./Fertig 6 DTA/dta_punctuation.txt",\
        "./Fertig 6 DTA/open_bracket.txt" ]

    output_hfst = 'any_punctuation.hfst'

    n = 10

    punctuation_characters = punctuation_characters_from_files(file_list)
    print("Punctuation Characters:", punctuation_characters)
    weighted_list = [(c, c, 0.0) for c in list(punctuation_characters)]

    print("Construct Transducer...")
    transducer = helper.transducer_from_list(weighted_list)
    transducer = hfst.HfstTransducer(transducer)

    #scaling_transducer = hfst.regex('?::1.0')
    #transducer.compose(scaling_transducer)

    transducer.optionalize()
    transducer.repeat_n(n)
    transducer.minimize()

    helper.save_transducer(output_hfst, transducer)
    print("Saved Transducer as", output_hfst)


if __name__ == '__main__':
    main()
