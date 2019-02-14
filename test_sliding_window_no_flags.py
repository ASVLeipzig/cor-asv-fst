import hfst
import helper 
import logging
from operator import itemgetter

from process_test_data import prepare_composition
import sliding_window_no_flags as sw

def build_model(use_composition=False):
    error_tr = helper.load_transducer('fst/error.hfst')
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
        return prepare_composition(window_acceptor, error_tr, 10, 100)
    else:
        return (error_tr, window_acceptor)

model = build_model(use_composition=True)

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

def p(string):
    global lexicon_fst, error_fst
    global composition
    print(string)
    tr = sw.process_string(string, model, max_window_size=2)
    for input_str, outputs in tr.extract_shortest_paths().items():
        for output_str, weight in outputs:
            print(output_str, weight, sep='\t')
    print()

p('Philo ophenvon Fndien dur<h Grieche nland bls')
p('„Aber Kind! Du redeſt ja ſchon den reinen Wahn-')
p('\ opf. Mir wurde plöblich fo klar, — jo ganz klar, daß')
p('ih denke. Aber was die ſelige Frau Geheimräth1n')
p('„Das fann ich niht, c’esl absolument impos-')
p('rend. In dem Augenbli> war 1hr niht wohl zu')
p('ür die fle ſich ſchlugen.“')
p('ſollte. Nur Über die Familien, wo man ſie einführen')
p('an der Hand führten <')
p('1 S2')
p('rath und ging auf und ab. Damit iſ es vorbei.')
p('erſhro>en haben, da ſoll Sie auch keine Seele finden.')
p('Cavalier.')
p('„Nuch ein Gru d, jetine Heimat zu verehren !“ be-')
p('— Limonade? — Knöpfen Sie den Pelz ein wenig')
p('vor dem Schuß, den er micht in den Rücken erhalten')
p('Sie aber ſah ihn mit demſelben ſtreigen Blick an, wie')
p('nein, bewahre! ſie hat ſozuſagen die Tendenz zur Che. “')
p('Ueberzeugung heraus ſprach ih. Glaube mir, ihr wollt')
p('„Man ſoll niht verlobt ſein“ meinte Gabriele')
p('Benno warf ihr durch ſeine Brille einen for)chenden')
