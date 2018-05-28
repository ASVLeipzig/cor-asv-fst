import hfst
import error_transducer as et

def main():

    #wb = et.load_transducer('wwmocr-fst/deu-ocr-wb.fst')
    #wb = et.load_transducer('wwmocr-fst/asse-big-nomor-tokerr.fst')
    #wb = et.load_transducer('wwmocr-fst/asse-big-nomor.fst')
    #wb = et.load_transducer('wwm/rules.fsm')
    #wb = et.load_transducer('wwm/lexicon.fsm')
    #wb = et.load_transducer('wwm/extended_lexicon_inverted.tropical')
    wb = et.load_transducer('result.fst')

    wb.n_best(50)

    results = wb.extract_paths()

    for input, outputs in results.items():
        print('%s:' % input)
        for output in outputs:
            print(' %s\t%f' % (output[0], output[1]))


    return 0




if __name__ == '__main__':
    main()
