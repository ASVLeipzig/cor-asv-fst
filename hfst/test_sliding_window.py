# encoding: utf-8

import pytest
import hfst

import sliding_window as sw


# prepare transducers and options
words_per_window = 3
result_num = 10
error_transducer, lexicon_transducer =\
    sw.load_transducers('transducers/max_error_3_context_23_dta.htsf',\
    'transducers/punctuation_transducer_dta.hfst',\
    'transducers/lexicon_transducer_dta.hfst')

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
    assert get_best_result(input_str) == '– 213 –'

