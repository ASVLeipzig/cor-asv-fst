# distutils: language = c++

from libcpp.string cimport string

cdef extern from "composition_cpp.h": 
    cdef cppclass Composition:
        Composition(string error_filename, string lexicon_filename, int nbest, float rejection_weight)
        string correct_string(string input_str)
        string correct_transducer_file(string input_transducer_filename)
        string correct_transducer_string(string input_transducer_string)
        int nbest

cdef class pyComposition: 

    cdef object error_filename
    cdef object lexicon_filename
    cdef int nbest
    cdef Composition* thisptr # hold a C++ instance
    
    def __cinit__(self, error_filename, lexicon_filename, int nbest, float rejection_weight):
        """Instantiate the OpenFST-based string correction:
        
        - read error transducer from the given filename,
        - read lexicon transducer from the given filename,
        - merge their symbol tables (and relabel arcs accordingly),
        - sort arcs.
        """
        self.error_filename = error_filename
        self.lexicon_filename = lexicon_filename
        self.nbest = nbest
        self.thisptr = new Composition(error_filename.encode('utf-8'), lexicon_filename.encode('utf-8'), nbest, rejection_weight)
    
    def __dealloc__(self):
        del self.thisptr
 
    def __repr__(self): 
        return "Composition[%s,%s,%s]" % (self.error_filename, self.lexicon_filename, self.nbest)

    def correct_string(self, input_str):
        """Correct input_str by 
        
        - creating a transducer with error_transducer's symbol table,
        - composing with error, lexicon and morphology transducer,
        - projecting the result transducer on the output,
        - removing epsilon transitions, and
        - searching for the nbest shortest=best paths.
        
        Write the result into a temporary file, and
        return the filename.
        """
        cdef string output_filename = self.thisptr.correct_string(input_str.encode('utf-8'))
        return output_filename.decode('utf-8')
    
    def correct_transducer_file(self, input_transducer_filename):
        """Correct input_transducer_filename by 
        
        - loading it into a transducer (and merging symbol tables),
        - composing with error, lexicon and morphology transducer,
        - projecting the result transducer on the output,
        - removing epsilon transitions, and
        - searching for the nbest shortest=best paths.
        
        Write the result into a temporary file, and
        return the filename.
        """
        cdef string output_filename = self.thisptr.correct_transducer_file(input_transducer_filename.encode('utf-8'))
        return output_filename.decode('utf-8')
    
    def correct_transducer_string(self, input_transducer_str):
        """Correct input_transducer_str by 
        
        - converting it into a transducer (and merging symbol tables),
        - composing with error, lexicon and morphology transducer,
        - projecting the result transducer on the output,
        - removing epsilon transitions, and
        - searching for the nbest shortest=best paths.
        
        Write the result into a temporary file, and
        return the filename.
        """
        cdef string output_filename = self.thisptr.correct_transducer_string(input_transducer_str.encode('utf-8'))
        return output_filename.decode('utf-8')

