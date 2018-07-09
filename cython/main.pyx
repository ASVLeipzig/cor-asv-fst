# distutils: language = c++

from composition import pyComposition;

def main():

    comp = pyComposition(b'/home/lena/Arbeit/Cython/Composition/max_error_3_context_23_dta.ofst', b'/home/lena/Arbeit/Cython/Composition/lexicon_transducer_dta.ofst', 10)
     
    print(comp)

    comp.compose(b'Haus')

main()
