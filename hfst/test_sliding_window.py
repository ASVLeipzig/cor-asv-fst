# encoding: utf-8

import pytest
import hfst

import sliding_window as sw


# prepare transducers and options
words_per_window = 3
result_num = 10
error_transducer, lexicon_transducer =\
    sw.load_transducers('transducers/max_error_3_context_23_dta.hfst',\
    'transducers/punctuation_transducer_dta.hfst',\
    'transducers/lexicon_transducer_dta.hfst',\
    'transducers/open_bracket_transducer_dta.hfst',\
    'transducers/close_bracket_transducer_dta.hfst')

lexicon_transducer.repeat_n(words_per_window)



def get_best_result(input_str):

    complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num)
    complete_output.n_best(1)
    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
    return list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', '')


def test_first_word_deleted():
    input_str = 'in Dir den Pater Medardus wieder zu erken.'
    assert get_best_result(input_str) == 'in Dir den Pater Medardus wieder zu erſten. '


def test_segmentation_errors():
    input_str = 'Philoſophenvon Fndicn dur<h Gricche nland bls'
    assert get_best_result(input_str) == 'Philoſophen von Jndien durch Griechenland bis '


def test_single_word():
    input_str = '213';
    assert get_best_result(input_str) == '213 '


def test_page_number():
    input_str = '– 213 –'
    assert get_best_result(input_str) == '– 213 – '


def test_quotation_mark_beginning():
    input_str = 'Graf: »Und ſind Sie Überraſcht durch die vor-'
    assert get_best_result(input_str) == 'Graf: »Und ſind Sie Überraſcht durch die vor—'


def test_quotation_mark_beginning():
    input_str = 'Steuergeſegebung und dergl. — erhalten haben.“ Ferner:'
    assert get_best_result(input_str) == 'Graf: »Und ſind Sie Überraſcht durch die vor—'


#Steuergeſegebung und dergl. — erhalten haben.“ Ferner:
#Steuergeſetzgebung und dergl. — erhalten haben.“ Ferner:
#Composition Time:  2.499004364013672
#Combination Time:  0.4121434688568115
#Composition Time:  3.355466365814209
#Combination Time:  0.5137724876403809
#Merge Time:  0.3937220573425293
#Steuer geſetzgebung und dergl. — erhalten haben.“ Ferner
#keller_heinrich01_1854_0083_018
#Graf: »Und ſind Sie Überraſcht durch die vor-
#Graf: »Und ſind Sie uͤberraſcht durch die vor—
#Composition Time:  2.341946840286255
#Combination Time:  0.40556979179382324
#Composition Time:  3.455350160598755
#Combination Time:  0.5091953277587891
#Merge Time:  0.31954240798950195
#Graf: »Und ſind Sie Überraſcht durch die vor—

