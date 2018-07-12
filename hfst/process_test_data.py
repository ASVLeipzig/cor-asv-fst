import sliding_window as sw
from os import listdir
import hfst
from composition import pyComposition


def get_txt_files(directory, model):
    return (f for f in listdir(directory) if f.endswith('.' + model + '.txt'))

def get_content(directory, file_generator):
    for x in file_generator:
        with open(directory + x) as f:
            yield [x.split('.')[0], f.read().strip()]

def create_dict(path, model):
    result_dict = {}
    files = get_txt_files(path, model)
    content = get_content(path, files)
    for file_id, line in content:
        result_dict[file_id] = line
    return result_dict

def write_txt_file(directory, model, name, string):

    return


def main():


    # prepare transducers

    words_per_window = 3
    result_num = 10
    error_transducer, lexicon_transducer =\
        sw.load_transducers('transducers/max_error_3_context_23_dta.hfst',\
        'transducers/punctuation_transducer_dta.hfst',\
        'transducers/lexicon_transducer_dta.hfst',\
        'transducers/open_bracket_transducer_dta.hfst',\
        'transducers/close_bracket_transducer_dta.hfst')

    lexicon_transducer.repeat_n(words_per_window)



    # prepare Composition Object

    error_filename = 'error.ofst'
    lexicon_filename = 'lexicon.ofst'
    error_filename_b = b'error.ofst'
    lexicon_filename_b = b'lexicon.ofst'

    for filename, fst in [(error_filename, error_transducer), (lexicon_filename, lexicon_transducer)]:
        out = hfst.HfstOutputStream(filename=filename, hfst_format=False, type=hfst.ImplementationType.TROPICAL_OPENFST_TYPE)
        out.write(fst)
        out.flush()
        out.close()

    composition = pyComposition(error_filename_b, lexicon_filename_b, result_num)


    # read and process test data

    path = '../../dta19-reduced/testdata/'

    gt_dict = create_dict(path, 'gt')
    fraktur4_dict = create_dict(path, 'Fraktur4')

    for key, value in list(fraktur4_dict.items()):#[10:20]:

        #input_str = value.strip(' \n\u000C')
        input_str = value

        print(key)
        print(value)
        print(gt_dict[key])

        complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num, composition)
        complete_output.n_best(1)
        complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)
        output_str = list(complete_paths.items())[0][1][0][0].replace('@_EPSILON_SYMBOL_@', '')

        print(output_str)
        print()

        with open(path + key + '.Fraktur4_corrected.txt', 'w') as f:
            f.write(output_str)


    return

if __name__ == '__main__':
    main()
