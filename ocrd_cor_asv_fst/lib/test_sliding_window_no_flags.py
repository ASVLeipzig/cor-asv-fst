import hfst
import helper 
import logging
from operator import itemgetter

from process_test_data import prepare_composition
import sliding_window_no_flags as sw

REJECTION_WEIGHT = 3

def build_model(use_composition=False):
    # error_tr = helper.load_transducer('fst/error.hfst')
    error_tr = helper.load_transducer('fst/error_st.fst')
    lexicon_tr = helper.load_transducer('fst/lexicon_transducer_dta.hfst')
    punctuation_tr = helper.load_transducer('fst/punctuation_transducer_dta.hfst')
    open_bracket_tr = helper.load_transducer('fst/open_bracket_transducer_dta.hfst')
    close_bracket_tr = helper.load_transducer('fst/close_bracket_transducer_dta.hfst')

    punctuation_tr.optionalize()
    open_bracket_tr.optionalize()
    close_bracket_tr.optionalize()

    single_token_acceptor = hfst.epsilon_fst()
    single_token_acceptor.concatenate(open_bracket_tr)
    single_token_acceptor.concatenate(lexicon_tr)
    single_token_acceptor.concatenate(punctuation_tr)
    single_token_acceptor.concatenate(close_bracket_tr)

    window_acceptor = single_token_acceptor.copy()
    window_acceptor.concatenate(hfst.fst(' '))
    window_acceptor.repeat_star()
    window_acceptor.concatenate(single_token_acceptor)

    if use_composition:
        return prepare_composition(window_acceptor, error_tr, 10, REJECTION_WEIGHT)
    else:
        return (error_tr, window_acceptor)

model = build_model(use_composition=True)

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def p(string, max_results=1):
    global lexicon_fst, error_fst
    global composition
    print(string)
    tr = sw.process_string(string, model, max_window_size=2, rejection_weight=REJECTION_WEIGHT)
    tr.n_best(max_results)
    paths = []
    for input_str, outputs in tr.extract_paths().items():
        for output_str, weight in outputs:
            paths.append((output_str, weight))
    for p in sorted(paths, key=itemgetter(1)):
        print(*p, sep='\t')
    print()

p('nem Makulaturbogen cinen Drukfehler mit Blei-')
p('daß Chen, die frúh gei<ſofſen wurden, anm')
p('Centauren, thre pelasgiſchen Kabiren verrathen uns')
