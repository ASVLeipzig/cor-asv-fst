import helper
import hfst


def get_digit_tuples():
    """Gives tuple of all pairs of identical numbers.
    This is used to replace the ('1', '1') transitions in the lexicon by
    all possible numbers."""

    return (('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5'),\
        ('6', '6'), ('7', '7'), ('8', '8'), ('9', '9'))


def main():
    """Read lexicon txt file (lexicon, punctuation, opening/closing
    brackets) created by process_dta_data and construnct hfst transducers."""

    print("Read Dictionary...")
    freq_list = helper.read_lexicon("dta_lexicon.txt")
    print("Construct Lexicon Transducer...")
    lexicon_transducer = helper.transducer_from_list(freq_list)
    # in the lexicon dict, numbers are counted as sequences of 1
    # thus, they are replaced by any possible number of the according length
    lexicon_transducer.substitute(('1', '1'), get_digit_tuples())
    lexicon_transducer = hfst.HfstTransducer(lexicon_transducer)
    helper.save_transducer('lexicon_transducer_dta.hfst', lexicon_transducer)

    helper.save_transducer_from_txt("dta_punctuation.txt", "punctuation_transducer_dta.hfst");
    helper.save_transducer_from_txt("open_bracket.txt", "open_bracket_transducer_dta.hfst");
    helper.save_transducer_from_txt("close_bracket.txt", "close_bracket_transducer_dta.hfst");

if __name__ == '__main__':
    main()
