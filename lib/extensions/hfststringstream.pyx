# distutils: language = c++

from libcpp.string cimport string
#from libhfst import HfstInputStream

# #from libcpp.sstream cimport istringstream # not predefined in Cython
# cdef extern from "<sstream>" namespace "std":
#     cdef cppclass istringstream:
#         istringstream()
#         istringstream(const string&) except +
# def string2HfstInputStream(string &str):
#     cdef istringstream istream = istringstream(str)
#     cdef HfstInputStream hstream = HfstInputStream(istream)
#     return hstream

cdef extern from "<hfst/HfstInputStream.h>" namespace "hfst":
    cdef cppclass HfstInputStream

cdef extern from "hfststringstream_cpp.h":
    HfstInputStream HfstInputStringStream(string str)

def string2HfstInputStream(s):
    return HfstInputStringStream(s)
