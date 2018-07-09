# distutils: language = c++

from libcpp.string cimport string

cdef extern from "composition_cpp.h": 
    cdef cppclass Composition:
        Composition(string error_file, string lexicon_file, int nbest)
        string compose(string input_str)
        string error_file
        string lexicon_file
        int nbest

cdef class pyComposition: 

    cdef string error_file
    cdef string lexicon_file
    cdef int nbest

    cdef Composition* thisptr # hold a C++ instance
    def __cinit__(self, string error_file, string lexicon_file, int nbest):
        self.error_file = error_file
        self.lexicon_file = lexicon_file
        self.nbest = nbest
        self.thisptr = new Composition(error_file, lexicon_file, nbest)
    def __dealloc__(self):
        del self.thisptr
 
    def __repr__(self): 
        return "Composition[%s,%s,%s]" % (self.thisptr.error_file, self.thisptr.lexicon_file, self.thisptr.nbest)

    def compose(self, string input_str):
        output_file = self.thisptr.compose(input_str)
        return output_file


    #cdef Test* thisptr # hold a C++ instance
    #def __cinit__(self, int test1):
    #    self.thisptr = new Test(test1)
    #def __dealloc__(self):
    #    del self.thisptr
 
    #def __add__(pyTest left, pyTest other):
    #    cdef Test t = left.thisptr.add(other.thisptr[0])
    #    cdef pyTest tt = pyTest(t.test1)
    #    return tt
    #def __sub__(pyTest left, pyTest other):
    #    cdef Test t = left.thisptr.sub(other.thisptr[0])
    #    cdef pyTest tt = pyTest(t.test1)
    #    return tt
 
    #def __repr__(self): 
    #    return "pyTest[%s]" % (self.thisptr.test1)
 
    #def returnFive(self):
    #    return self.thisptr.returnFive()

    #def returnThree(self):
    #    return self.thisptr.returnThree()

    #def readFst(self):
    #    return self.thisptr.readFst()

    #def printMe(self):
    #    return "hello world"





#public: 
#
#    Composition(const string &error_file, const string &lexicon_file, int nbest);
#    ~Composition();
#
#    string compose(const string input_str);
#
#private:
#
#    SVF create_input_transducer(string word, const SymbolTable* table);
#    SVF eager_compose(SVF *input1, SVF *input2, SVF *input3, string word);
#    ComposeFst<StdArc> lazy_compose(SVF *input1, SVF *input2, SVF *input3, string word);
#    SVF compose_and_search(SVF *input1, SVF *input2, SVF *input3, string word, bool lazy, int nbest);
#
#    int nbest; 
#
#    SVF *error_transducer;
#    SVF *lexicon_transducer;




