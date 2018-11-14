#include "composition_cpp.h"

//#define COMPOSITION_TIMES 1 /* print cputime measurements on stderr? */

/* Instantiate the OpenFST-based string correction:
   - read error transducer from the given filename,
   - read lexicon transducer from the given filename,
   - merge their symbol tables (and relabel arcs accordingly),
   - sort arcs.
 */ 
Composition::Composition(const string &error_file, const string &lexicon_file, int nbest, float rejection_weight)
  : nbest(nbest),
    rejection_weight(rejection_weight),
    error_transducer(StdVectorFst::Read(error_file)),
    lexicon_transducer(StdVectorFst::Read(lexicon_file)),
    morphology_transducer(NULL) {

    vector<SVF*> arc_sort_required = {
        error_transducer,
        lexicon_transducer // really necessary on both sides?
        //, morphology_transducer
    };

    // merge and relabel symbol tables
    bool relabel;

    const SymbolTable output_symbols = *(error_transducer->OutputSymbols());
    const SymbolTable input_symbols = *(lexicon_transducer->InputSymbols());

    this->symbol_table = MergeSymbolTable(output_symbols, input_symbols, &relabel);

    if (relabel) {
      Relabel(lexicon_transducer, this->symbol_table, this->symbol_table);
    }

    lexicon_transducer->SetOutputSymbols(this->symbol_table);
    lexicon_transducer->SetInputSymbols(this->symbol_table);

    error_transducer->SetOutputSymbols(this->symbol_table);
    error_transducer->SetInputSymbols(this->symbol_table);

    /*
    // write symbol table to file
    this->symbol_table.WriteText("symbol_table.txt"); 
    this->symbol_table.Write("symbol_table.bin"); 
    */
    
    // perfom arc sort
    for (int i = 0; i < std::end(arc_sort_required) - std::begin(arc_sort_required); i++) {

        ArcSort(arc_sort_required[i], StdOLabelCompare());
        ArcSort(arc_sort_required[i], StdILabelCompare());

        // EpsNormalize?
        
        //const SVF* to_convert = *arc_sort_required[i];
        //arc_sort_required[i] = Convert(*to_convert, "arc_lookahead");
    }

    FlagRegister<string> *stringvar_register = FlagRegister<string>::GetRegister();
    if (not (stringvar_register->SetFlag(string(""), &FLAGS_fst_field_separator))) // prevent space and tab (keep only implicit newline)
      throw std::runtime_error("cannot reset fst_field_separator");

    /* to re-use this for each window of each line, we want this to be a member
       but we cannot just declare a private StringCompiler<StdArc> member
       due to the restrictive definition in fst/string.h, which deletes operator=
       and forbids copy initialization via explicit.
       But we cannot provide a default initialization for it, we strictly
       depend on the merged symbol table calculated in the constructor.
       So instead we have to do this with pointers.
    */
    //this->string_compiler(SYMBOL, &this->symbol_table); // UTF8
    this->string_compiler = new StringCompiler<StdArc>(SYMBOL, this->symbol_table); // UTF8 does not allow for multi-char symbols

    /* same here: we want a WeightConvertMapper for a Converter that depends on 
       rejection_weight given in the constructor. 
     */
    this->weight_mapper = new WeightConvertMapper<StdArc, StdArc, RejectionWeight>((RejectionWeight(W(this->rejection_weight))));
    
}


Composition::~Composition() {
  delete this->string_compiler;
  delete this->weight_mapper;
}


/* Correct input_transducer_filename by 
   - loading it into a transducer (and merging symbol tables),
   - composing with error, lexicon and morphology transducer,
   - projecting the result transducer on the output,
   - removing epsilon transitions, and
   - searching for the nbest shortest=best paths.
   Write the result into a temporary file, and
   return the filename.
*/
string Composition::correct_transducer_file(const string input_transducer_filename) {
  std::unique_ptr<SVF> input_transducer(StdVectorFst::Read(input_transducer_filename));
  if (input_transducer == nullptr) {
    // libfst will already print error
    throw std::runtime_error("cannot read input transducer file");
  }

  // relabel input transducer with big symbol table
  Relabel(input_transducer.get(), this->symbol_table, this->symbol_table);

  input_transducer->SetInputSymbols(this->symbol_table);
  input_transducer->SetOutputSymbols(this->symbol_table);

  std::unique_ptr<SVF> nbest_transducer =
    compose_and_search(input_transducer.get(),
                       false); // lazy
    
  //string filename = string("output/") +  input_transducer_filename + string(".fst");
  // FIXME: use StdVectorFst::FstToString() here and wrap with HfstInputStream(std::istringstream) in Cython
  string filename = std::tmpnam(NULL);
  nbest_transducer->Write(filename);
  nbest_transducer.reset();
  input_transducer.reset();
    
  return filename;
}

/* Correct input_transducer_str by 
   - converting it into a transducer (and merging symbol tables),
   - composing with error, lexicon and morphology transducer,
   - projecting the result transducer on the output,
   - removing epsilon transitions, and
   - searching for the nbest shortest=best paths.
   Write the result into a temporary file, and
   return the filename.
*/
string Composition::correct_transducer_string(const string input_transducer_str) {
  // there is a weird bug in gcc here: it fails to grasp the VectorFst<StdArc> template (hence the cast)
  std::unique_ptr<SVF> input_transducer((SVF*)StringToFst<StdArc>(input_transducer_str));
  if (input_transducer == nullptr) {
    // libfst will already print error
    throw std::runtime_error("cannot read input transducer string");
  }

  // relabel input transducer with big symbol table
  Relabel(input_transducer.get(), this->symbol_table, this->symbol_table);

  input_transducer->SetInputSymbols(this->symbol_table);
  input_transducer->SetOutputSymbols(this->symbol_table);

  std::unique_ptr<SVF> nbest_transducer =
    compose_and_search(input_transducer.get(),
                       false); // lazy

  // FIXME: use StdVectorFst::FstToString() here and wrap with HfstInputStream(std::istringstream) in Cython
  string filename = std::tmpnam(NULL);
  nbest_transducer->Write(filename);
  nbest_transducer.reset();
  input_transducer.reset();
  
  return filename;
}


/* Correct input_str by 
   - creating a transducer with error_transducer's symbol table,
   - composing with error, lexicon and morphology transducer,
   - projecting the result transducer on the output,
   - removing epsilon transitions, and
   - searching for the nbest shortest=best paths.
   Write the result into a temporary file, and
   return the filename.
*/
string Composition::correct_string(const string input_str) {

  std::unique_ptr<SVF> nbest_transducer =
    compose_and_search(input_str,
                       false); // lazy
    
  //string filename = string("output/") +  input_str + string(".fst");
  // FIXME: use StdVectorFst::FstToString() here and wrap with HfstInputStream(std::istringstream) in Cython
  string filename = std::tmpnam(NULL);
  nbest_transducer->Write(filename);
  nbest_transducer.reset();

  return filename;
}


/* Correct input_transducer by 
   - composing all inputs,
   - projecting the result transducer on the output,
   - removing epsilon transitions, and
   - searching for the nbest shortest=best paths.
*/
std::unique_ptr<SVF> Composition::compose_and_search(SVF *input_transducer,
                                                     bool lazy) {

  std::unique_ptr<SVF> nbest_transducer = std::make_unique<SVF>();

  //int32 nshortest = 1;
  int32 nshortest = this->nbest;
  bool unique = true;
  bool first_path = false;
  static StdArc::Weight weight_threshold = StdArc::Weight::Zero();
  static StdArc::StateId state_threshold = kNoStateId;
  static std::vector<StdArc::Weight> distance;

  // create arc filter :

  static AnyArcFilter<StdArc> arc_filter;
  //LookAheadComposeFilter<StdArc> arc_filter;

  //LAM* lam1 = new LAM(A, MATCH_OUTPUT);
  //LAM* lam2 = new LAM(B, MATCH_INPUT);
  //
  //LCF* laf1 = new LCF( A, B, lam1, lam2);

  // create queue:

  //AutoQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
  //NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>* state_queue =
  //    new NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>(distance);
  static NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(distance);

  // create shortest path options:

  //const ShortestPathOptions<StdArc, AutoQueue<StdArc::StateId>, AnyArcFilter<StdArc>> opts(
  static const ShortestPathOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc>>
    opts(&state_queue, arc_filter, nshortest, unique, false, kDelta, first_path, weight_threshold, state_threshold);


  //if (lazy) {

  //    std::unique_ptr<StdComposeFst> composed = lazy_compose(input_str);

  //    //float compose_time = get_cpu_time();
  //    //cout << "Composition: " << compose_time << endl;

  //    //ShortestPath(*composed, nbest_transducer.get(), this->nbest);
  //    ShortestPath(*composed, nbest_transducer.get(), &distance, opts);
  //    composed.reset();

  //    //float total_time = get_cpu_time();
  //    //float search_time = total_time - compose_time;
  //    //cout << "Search: " << search_time << endl;
  //    //cout << "Total time: " << total_time << endl;

  //}
  //else {

  std::unique_ptr<SVF> composed = eager_compose(input_transducer);

#ifdef COMPOSITIN_TIMES
  float compose_time = get_cpu_time();
#endif

  // perform output projection, since unique option in shortest path
  // search requires automaton/acceptor as input
  if (unique) {
    Project(composed.get(), PROJECT_OUTPUT);
    RmEpsilon(composed.get());
  }

  ShortestPath(*composed, nbest_transducer.get(), &distance, opts);
  composed.reset();

#ifdef COMPOSITIN_TIMES
  float search_time = get_cpu_time();
  cerr << "Search Time: " << (search_time - compose_time) << endl;
#endif
  //}

  return std::move(nbest_transducer);
}


/* Correct input_str by 
   - creating a transducer for input_str from input1's symbol table,
   - composing all inputs,
   - projecting the result transducer on the output,
   - removing epsilon transitions, and
   - searching for the nbest shortest=best paths.
*/
std::unique_ptr<SVF> Composition::compose_and_search(string input_str,
                                                     bool lazy) {

  std::unique_ptr<SVF> nbest_transducer = std::make_unique<SVF>();

  //int32 nshortest = 1;
  int32 nshortest = this->nbest;
  bool unique = true;
  bool first_path = false;
  static StdArc::Weight weight_threshold = StdArc::Weight::Zero();
  static StdArc::StateId state_threshold = kNoStateId;
  static std::vector<StdArc::Weight> distance;

  // create arc filter 

  static AnyArcFilter<StdArc> arc_filter;
  //LookAheadComposeFilter<StdArc> arc_filter;

  //LAM* lam1 = new LAM(A, MATCH_OUTPUT);
  //LAM* lam2 = new LAM(B, MATCH_INPUT);
  //
  //LCF* laf1 = new LCF( A, B, lam1, lam2);


  // create queue

  //AutoQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
  //NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>* state_queue =
  //    new NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>(distance);
  static NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(distance);

  // create shortest path options

  //const ShortestPathOptions<StdArc, AutoQueue<StdArc::StateId>, AnyArcFilter<StdArc>> opts(
  static const ShortestPathOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc>>
    opts(&state_queue, arc_filter, nshortest, unique, false, kDelta, first_path, weight_threshold, state_threshold);

#ifdef COMPOSITIN_TIMES
  float time_started = get_cpu_time();
#endif

  if (lazy) {

    std::unique_ptr<StdComposeFst> composed = lazy_compose(input_str);

#ifdef COMPOSITIN_TIMES
    float compose_time = get_cpu_time();
    cerr << "Compose Time: " << (compose_time-time_started) << endl;
#endif

    //ShortestPath(composed.get(), nbest_transducer, this->nbest);
    ShortestPath(*composed, nbest_transducer.get(), &distance, opts);
    composed.reset();

#ifdef COMPOSITIN_TIMES
    float search_time = get_cpu_time();
    cerr << "Search Time: " << (search_time - compose_time) << endl;
#endif
  } else {

    std::unique_ptr<SVF> composed = eager_compose(input_str);

#ifdef COMPOSITIN_TIMES
    float compose_time = get_cpu_time();
    cerr << "Compose Time: " << (compose_time-time_started) << endl;
#endif

    //ShortestPath(*composed, nbest_transducer.get(), nbest);

    // perform output projection, since unique option in shortest path
    // search requires automaton/acceptor as input
    if (unique) {
      Project(composed.get(), PROJECT_OUTPUT);
      RmEpsilon(composed.get());
    }

    ShortestPath(*composed, nbest_transducer.get(), &distance, opts);
    composed.reset();

#ifdef COMPOSITIN_TIMES
    float search_time = get_cpu_time();
    cerr << "Search Time: " << (search_time - compose_time) << endl;
#endif
  }

#ifdef COMPOSITIN_TIMES
  /*
  if (getrusage(RUSAGE_SELF, &this->usage) < 0) {
    std::perror("cannot get usage statistics");
    // exit(1);
  } else {
    
    // maximum resident set size in kB
    std::cout << "RAM usage: " << this->usage.ru_maxrss << endl;
  }
  */
#endif

  return std::move(nbest_transducer);
}


/* Compose all inputs lazily, 
   creating a transducer for input_str
   from input1's symbol table first.
*/
std::unique_ptr<StdComposeFst> Composition::lazy_compose(string input_str) {

  SVF *input1 = this->error_transducer;
  SVF *input2 = this->lexicon_transducer;
  SVF *input3 = this->morphology_transducer;

#ifdef COMPOSITIN_TIMES
  float time_started = get_cpu_time();
#endif
    
  const SymbolTable table = *(input1->OutputSymbols());
    
  std::unique_ptr<SVF> input_transducer = create_input_transducer(input_str);
  ArcSort(input_transducer.get(), StdOLabelCompare()); // really necessary on both sides?

#ifdef COMPOSITIN_TIMES
  float time_created = get_cpu_time();
  cerr << "Create Time: " << (time_created-time_started) << endl;
#endif
    
  //input_transducer->Write(string("input_") + input_str + string(".fst"));

  //ArcSort(input1, StdOLabelCompare());
  //ArcSort(input2, StdILabelCompare());

  //COpts* opts = new COpts();
  static COpts opts;

  //ComposeFst<StdArc>* delayed_result = new ComposeFst<StdArc>(*input1, *input2, opts);
  std::unique_ptr<StdComposeFst> delayed_result = std::make_unique<StdComposeFst>(*input1, *input2, opts);

  if (input3) {
    //ArcSort(input3, StdILabelCompare());
    //delayed_result2 = new ComposeFst<StdArc>(delayed_result, *input3, opts);
    delayed_result.reset(new StdComposeFst(*delayed_result, *input3, opts));
  }
    
  //delayed_result2 = new ComposeFst<StdArc>(input_transducer, (input3 ? *delayed_result2 : delayed_result), opts);
  delayed_result.reset(new StdComposeFst(*input_transducer, *delayed_result, opts));
    
#ifdef COMPOSITIN_TIMES
  float time_composed = get_cpu_time();
  cerr << "Compose Time: " << (time_composed - time_created) << endl;
#endif

  /* FIXME does not work -- maybe we need a delayed backoff variant?
     so the caller must do backoff herself (union with reweighted input, rmepsilon, determinize)!
  std::unique_ptr<SVF> output_transducer = backoff_result(input_transducer.get(), delayed_result.get());
  input_transducer.reset();
  delayed_result.reset();
  
  return std::move(output_transducer);
  */

  return std::move(delayed_result);
}


/* Compose all inputs eagerly.
*/
std::unique_ptr<SVF> Composition::eager_compose(SVF *input_transducer) {

  SVF *input1 = this->error_transducer;
  SVF *input2 = this->lexicon_transducer;
  SVF *input3 = this->morphology_transducer;

#ifdef COMPOSITIN_TIMES
  float time_started = get_cpu_time();
#endif
    
  ArcSort(input_transducer, StdOLabelCompare()); // really necessary on both sides?

#ifdef COMPOSITIN_TIMES
  float time_created = get_cpu_time();
  cerr << "Create Time: " << (time_created-time_started) << endl;
#endif

  std::unique_ptr<SVF> result = std::make_unique<SVF>();

  Compose(*input_transducer, *input1, result.get());
 
  if (input2) {
    //ArcSort(result, StdOLabelCompare());
    Compose(*result, *input2, result.get());
  }

  if (input3) {
    //ArcSort(result, StdOLabelCompare());
    Compose(*result, *input3, result.get());
  }

#ifdef COMPOSITIN_TIMES
  float time_composed = get_cpu_time();
  cerr << "Compose Time: " << (time_composed - time_created) << endl;
#endif
  
  std::unique_ptr<SVF> output_transducer = backoff_result(input_transducer, result.get());
  result.reset();

  return std::move(output_transducer);

}


/* Compose all inputs earerly, 
   creating a transducer for input_str
   from input1's symbol table first.
*/
std::unique_ptr<SVF> Composition::eager_compose(string input_str) {

  SVF *input1 = this->error_transducer;
  SVF *input2 = this->lexicon_transducer;
  SVF *input3 = this->morphology_transducer;

#ifdef COMPOSITIN_TIMES
  float time_started = get_cpu_time();
#endif
    
  const SymbolTable table = *(input1->OutputSymbols());
    
  std::unique_ptr<SVF> input_transducer = create_input_transducer(input_str);
  ArcSort(input_transducer.get(), StdOLabelCompare()); // really necessary on both sides?
    
#ifdef COMPOSITIN_TIMES
  float time_created = get_cpu_time();
  cerr << "Create Time: " << (time_created-time_started) << endl;
#endif

  //input_transducer->Write(string("input_") + input_str + string(".fst"));

  std::unique_ptr<SVF> result = std::make_unique<SVF>();

  Compose(*input_transducer, *input1, result.get());

  if (input2) {
    //ArcSort(result, StdOLabelCompare());
    Compose(*result, *input2, result.get());
  }

  if (input3) {
    //ArcSort(result, StdOLabelCompare());
    Compose(*result, *input3, result.get());
  }

#ifdef COMPOSITIN_TIMES
  float time_composed = get_cpu_time();
  cerr << "Compose Time: " << (time_composed - time_created) << endl;
#endif

  std::unique_ptr<SVF> output_transducer = backoff_result(input_transducer.get(), result.get());
  input_transducer.reset();
  result.reset();
  
  return std::move(output_transducer);
}

std::unique_ptr<SVF> Composition::backoff_result(SVF *input_transducer, SVF *output_transducer) {

#ifdef COMPOSITIN_TIMES
  float time_started = get_cpu_time();
#endif
  
  // apply rejection_weight to input:
  ArcMap(input_transducer, this->weight_mapper);

  // maybe we should also Prune(&result, StdArc::Weight 1.5) now? Intuitively, no paths worse than worst rejection path must remain
  // if we do, then determinize should be given the same weight threshold below
    
  // disjoin result with input (acts as a rejection threshold):
  Union(output_transducer, *input_transducer); // faster in that direction

#ifdef COMPOSITIN_TIMES
  float time_backoff = get_cpu_time();
  cerr << "Backoff Time: " << (time_backoff - time_started) << endl;
#endif
  
  // remove epsilon-epsilon transitions:
  RmEpsilon(output_transducer);
    
  // connect (make coaccessible) not necessary as included by RmEpsilon already: Connect(&delayed_result3);

  // disambiguate?

  // determinize and prune (in preparation of nbest search):
  static EncodeMapper<StdArc> codec(kEncodeLabels, ENCODE); // |kEncodeWeights
  Encode(output_transducer, &codec);
  std::unique_ptr<SVF> result = std::make_unique<SVF>();
  Determinize(*output_transducer, result.get(), DeterminizeOptions<StdArc>(DETERMINIZE_DISAMBIGUATE));
  Decode(result.get(), codec);
    
#ifdef COMPOSITIN_TIMES
  float time_determinized = get_cpu_time();
  cerr << "Determinize Time: " << (time_determinized - time_composed) << endl;
#endif
    
  return std::move(result);
}

/* Convert the input string into an input transducer
   using the given symbol table. 
   Expects input to be encoded like the symbols table,
   and pre-tokenized (a newline-separated list).
*/
std::unique_ptr<SVF> Composition::create_input_transducer(string input_str) {

  std::unique_ptr<SVF> input_transducer = std::make_unique<SVF>();
  //if (not string_compiler(input_str, input_transducer))
  if (not (*string_compiler)(input_str, input_transducer.get()))
    throw std::invalid_argument("input string incompatible with given symbol table: " + input_str);

  return std::move(input_transducer);
}


float Composition::get_cpu_time() {

  if (getrusage(RUSAGE_SELF, &this->usage) < 0) {
    std::perror("cannot get usage statistics");
    // exit(1);
    return -1;
  } else {
    
    return this->usage.ru_utime.tv_sec + this->usage.ru_stime.tv_sec +
      1e-6*this->usage.ru_utime.tv_usec + 1e-6*this->usage.ru_stime.tv_usec;
    
  }
}
