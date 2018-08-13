#ifndef TEST_H
#define TEST_H

#include <fst/fstlib.h>
#include <string>
#include <codecvt>
#include <locale>

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

    Composition(const string &error_file, const string &lexicon_file, int nbest);
    ~Composition();

    string compose(const string input_str);
    string compose_file(const string input_str);

    int nbest; 

    string error_file;
    string lexicon_file;

private:

    SymbolTable symbol_table;

    SVF create_input_transducer(string word, const SymbolTable* table);

    SVF eager_compose(SVF *input1, SVF *input2, SVF *input3, string word);
    SVF eager_compose(SVF *input1, SVF *input2, SVF *input3, SVF *input_transducer);

    ComposeFst<StdArc> lazy_compose(SVF *input1, SVF *input2, SVF *input3, string word);
    SVF compose_and_search(SVF *input1, SVF *input2, SVF *input3, string word, bool lazy, int nbest);
    SVF compose_and_search(SVF *input1, SVF *input2, SVF *input3, SVF *input_transducer, bool lazy, int nbest);

    SVF *error_transducer;
    SVF *lexicon_transducer;

};
#endif
