# encoding: utf-8

import pytest
import hfst

import sliding_window as sw


# prepare transducers and options
words_per_window = 3
composition_depth = 1
result_num = 10
flag_encoder = sw.FlagEncoder()

error_transducer, lexicon_transducer =\
    sw.load_transducers_bracket('transducers/max_error_3_context_23_dta.hfst',\
    'transducers/punctuation_transducer_dta.hfst',\
    'transducers/lexicon_transducer_dta.hfst',\
    'transducers/open_bracket_transducer_dta.hfst',\
    'transducers/close_bracket_transducer_dta.hfst',
    flag_encoder,\
    composition_depth = composition_depth,\
    words_per_window = words_per_window)

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

composition = sw.pyComposition(error_filename_b, lexicon_filename_b, result_num)
print(composition)


def get_best_result(input_str):

    complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, flag_encoder, result_num, composition)
    complete_output.n_best(1)

    complete_output = sw.remove_flags(hfst.HfstBasicTransducer(complete_output), flag_encoder)
    complete_output = hfst.HfstTransducer(complete_output)

    complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)

    return list(complete_paths.items())[0][1][0][0].replace(hfst.EPSILON, '')


def test_first_word_deleted():
    input_str = 'in Dir den Pater Medardus wieder zu '
    assert get_best_result(input_str) == 'in Dir den Pater Medardus wieder zu '

def test_segmentation_errors():
    input_str = 'Philoſophenvon Fndicn dur<h Gricche nland bis'
    assert get_best_result(input_str) == 'Philoſophen von Jndien durch Griechenland bis '


def test_single_word():
    input_str = '213';
    assert get_best_result(input_str) == '213 '


def test_page_number():
    input_str = '– 213 –'
    assert get_best_result(input_str) == '– 213 – '


def test_quotation_mark_beginning():
    input_str = 'Graf: »Und ſind Sie Überraſcht durch die vor-'
    assert get_best_result(input_str) == 'Graf: »Und ſind Sie Überraſcht durch die vor— '

def test_quotation_mark_beginning():
    # this tests fails, because the quatation marks are replaced by other
    # quotation marks; replacing punctuation by other punctuation seems to
    # be too inexpensive
    input_str = 'heit Anderer?« fragte ich lächelnd. »Mehr als jedes'
    assert get_best_result(input_str) == 'heit Anderer?« fragte ich lächelnd. »Mehr als jedes '



