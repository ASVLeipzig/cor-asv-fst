import sliding_window as sw
from os import listdir
import hfst


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

    words_per_window = 3
    result_num = 10
    error_transducer, lexicon_transducer = sw.load_transducers('transducers/max_error_3_context_23_dta.htsf',\
        'transducers/punctuation_transducer_dta.hfst',\
        'transducers/lexicon_transducer_dta.hfst')

    lexicon_transducer.repeat_n(words_per_window)


    path = '../dta19-reduced/testdata/'

    gt_dict = create_dict(path, 'gt')
    fraktur4_dict = create_dict(path, 'Fraktur4')

    for key, value in list(fraktur4_dict.items())[43:]:#[10:20]:

        #input_str = value.strip(' \n\u000C')
        input_str = value

        print(key)
        print(value)
        print(gt_dict[key])



        complete_output = sw.window_size_1_2(input_str, error_transducer, lexicon_transducer, result_num)

        complete_output.n_best(200)
        complete_paths = hfst.HfstTransducer(complete_output).extract_paths(max_number=1, max_cycles=0)

        for input, outputs in complete_paths.items():
            #print('%s:' % input.replace('@_EPSILON_SYMBOL_@', '□'))
            for output in outputs:
                #print(' %s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))
               print('%s\t%f' % (output[0].replace('@_EPSILON_SYMBOL_@', ''), output[1]))

        print()



    return

if __name__ == '__main__':
    main()
