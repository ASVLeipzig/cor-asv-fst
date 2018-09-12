import hfst

def save_transducer(filename, transducer):
    ostr = hfst.HfstOutputStream(filename=filename)
    ostr.write(transducer)
    ostr.flush()
    ostr.close()


def load_transducer(filename):
    transducer = None
    istr = hfst.HfstInputStream(filename)
    while not istr.is_eof():
        transducer = istr.read()
    istr.close()

    return transducer
