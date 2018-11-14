#ifndef COMPOSITION_H
#define COMPOSITION_H

#include <fst/fstlib.h>
#include <string>
#include <cstdio>
#include <codecvt>
#include <locale>

#include <sys/time.h>
#include <sys/resource.h>
#include <ctime>

using namespace fst;
using namespace std;

typedef StdVectorFst SVF;
typedef StdArc A;
typedef A::StateId S;
typedef A::Weight W;
typedef Matcher< Fst<A> > M;
typedef SequenceComposeFilter<M> F;
typedef GenericComposeStateTable<A, F::FilterState> T;
typedef ComposeFstOptions<A, M, F, T> COpts;

//typedef ArcLookAheadMatcher< SortedMatcher<SVF> > LAM;
//typedef SequenceComposeFilter< LAM, LAM > SCF;
//typedef LookAheadComposeFilter< SCF, LAM, LAM, MATCH_BOTH> LCF;

 
class Composition { 

public: 

    Composition(const string &error_file, const string &lexicon_file, int nbest, float rejection_weight);
    ~Composition();

    string correct_string(const string input_str);
    string correct_transducer_file(const string input_transducer_filename);
    string correct_transducer_string(const string input_transducer_str);

    int nbest; 
    float rejection_weight; 

private:

    class RejectionWeight {
    public:
      explicit RejectionWeight(const W &r_ = W::One()) : r(r_) {}
      W operator()(W w) const {
        return Times<float>(w, r);
      }
    private:
      W r;
    };

    WeightConvertMapper<StdArc, StdArc, RejectionWeight> *weight_mapper; // order: depends on rejection weight
    
    SVF *error_transducer;
    SVF *lexicon_transducer;
    SVF *morphology_transducer;
    SymbolTable *symbol_table; // order: depends on error and lexicon transducer

    //StringCompiler<StdArc> string_compiler;
    StringCompiler<StdArc> *string_compiler; // order: depends on symbol table
    std::unique_ptr<SVF> create_input_transducer(string input_str);

    std::unique_ptr<SVF> backoff_result(SVF *input_transducer, SVF *output_transducer);
    
    std::unique_ptr<SVF> eager_compose(string input_str);
    std::unique_ptr<SVF> eager_compose(SVF *input_transducer);
    std::unique_ptr<StdComposeFst> lazy_compose(string input_str);
    
    std::unique_ptr<SVF> compose_and_search(string input_str, bool lazy);
    std::unique_ptr<SVF> compose_and_search(SVF *input_transducer, bool lazy);

    struct rusage usage;
    float get_cpu_time();

};
#endif
