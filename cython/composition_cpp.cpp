#include "composition_cpp.h"
 
 
Composition::Composition(const string &error_file, const string &lexicon_file, int nbest) {

    // read transducer files

    SVF *error_transducer = StdVectorFst::Read(error_file);
    SVF *lexicon_transducer = StdVectorFst::Read(lexicon_file);

    vector<SVF*> arc_sort_required = {
        error_transducer,
        lexicon_transducer
    };

    // merge and relabel symbol tables

    bool relabel;

    const SymbolTable output_symbols = *(error_transducer->OutputSymbols());
    const SymbolTable input_symbols = *(lexicon_transducer->InputSymbols());

    SymbolTable new_symbol_table = *(MergeSymbolTable(output_symbols, input_symbols, &relabel));

    if (relabel) {
      Relabel(lexicon_transducer, &new_symbol_table, &new_symbol_table);
    }

    lexicon_transducer->SetOutputSymbols(&new_symbol_table);
    lexicon_transducer->SetInputSymbols(&new_symbol_table);

    error_transducer->SetOutputSymbols(&new_symbol_table);
    error_transducer->SetInputSymbols(&new_symbol_table);


    // write symbol table to file

    new_symbol_table.WriteText("symbol_table.txt"); 
    new_symbol_table.Write("symbol_table.bin"); 
    

    // perfom arc sort

    for (int i = 0; i < std::end(arc_sort_required) - std::begin(arc_sort_required); i++) {

        ArcSort(arc_sort_required[i], StdOLabelCompare());
        ArcSort(arc_sort_required[i], StdILabelCompare());

        //const SVF* to_convert = *arc_sort_required[i];
        //arc_sort_required[i] = Convert(*to_convert, "arc_lookahead");

    }

    
    // set class variables

    this->nbest = nbest; 
    this->error_file = error_file;
    this->lexicon_file = lexicon_file;
    this->error_transducer = error_transducer;
    this->lexicon_transducer = lexicon_transducer;
    this->symbol_table = new_symbol_table;

}


Composition::~Composition() { }


string Composition::compose_file(const string input_file) {

    SVF *input_transducer = StdVectorFst::Read(string("input/") + input_file + string(".hfst"));

    // relabel input transducer with big symbol table

    Relabel(input_transducer, &(this->symbol_table), &(this->symbol_table));

    input_transducer->SetInputSymbols(&(this->symbol_table));
    input_transducer->SetOutputSymbols(&(this->symbol_table));

    SVF nbest_transducer;

    nbest_transducer = compose_and_search(
        this->error_transducer,
        this->lexicon_transducer,
        NULL, // morphology
        input_transducer,
        false, // lazy
        this->nbest);
    
    string filename = string("output/") +  input_file + string(".fst");

    nbest_transducer.Write(filename);

    delete input_transducer;

    return filename;
}


string Composition::compose(const string input_str) {

    SVF nbest_transducer;

    nbest_transducer = compose_and_search(
        this->error_transducer,
        this->lexicon_transducer,
        NULL, // morphology
        input_str,
        false, // lazy
        this->nbest);
    
    string filename = string("output/") +  input_str + string(".fst");

    nbest_transducer.Write(filename);

    return filename;
}


SVF Composition::compose_and_search(
    SVF *input1,
    SVF *input2,
    SVF *input3,
    SVF *input_transducer,
    bool lazy,
    int nbest
    ) {

    SVF nbest_transducer;


    //int32 nshortest = 1;
    int32 nshortest = nbest;
    bool unique = true;
    bool first_path = false;
    StdArc::Weight weight_threshold = StdArc::Weight::Zero();
    StdArc::StateId state_threshold = kNoStateId;
    std::vector<StdArc::Weight> distance;

    // create arc filter 

    AnyArcFilter<StdArc> arc_filter;
    //LookAheadComposeFilter<StdArc> arc_filter;

    //LAM* lam1 = new LAM(A, MATCH_OUTPUT);
    //LAM* lam2 = new LAM(B, MATCH_INPUT);
    //
    //LCF* laf1 = new LCF( A, B, lam1, lam2);


    // create queue

    //AutoQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
    //NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>* state_queue =
    //    new NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>(distance);
    NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(distance);

    // create shortest path options

    //const ShortestPathOptions<StdArc, AutoQueue<StdArc::StateId>, AnyArcFilter<StdArc>> opts(
    const ShortestPathOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc>>
        opts(&state_queue, arc_filter, nshortest, unique, false, kDelta, first_path, weight_threshold, state_threshold);


    //if (lazy) {

    //    ComposeFst<StdArc> composed = lazy_compose(input1, input2, input3, word);

    //    //float compose_time = get_cpu_time();
    //    //cout << "Composition: " << compose_time << endl;


    //    //ShortestPath(composed, &nbest_transducer, nbest);

    //    ShortestPath(composed, &nbest_transducer, &distance, opts);

    //    //float total_time = get_cpu_time();
    //    //float search_time = total_time - compose_time;
    //    //cout << "Search: " << search_time << endl;
    //    //cout << "Total time: " << total_time << endl;

    //}
    //else {

    SVF composed = eager_compose(input1, input2, input3, input_transducer);


    // perform output projection, since unique option in shortest path
    // search requires automaton/acceptor as input
    if (unique) {
        Project(&composed, PROJECT_OUTPUT);
        RmEpsilon(&composed);
    }

    ShortestPath(composed, &nbest_transducer, &distance, opts);

    //}

    return nbest_transducer;
}






 
SVF Composition::compose_and_search(
    SVF *input1,
    SVF *input2,
    SVF *input3,
    string word,
    bool lazy,
    int nbest
    ) {

    SVF nbest_transducer;


    //int32 nshortest = 1;
    int32 nshortest = nbest;
    bool unique = true;
    bool first_path = false;
    StdArc::Weight weight_threshold = StdArc::Weight::Zero();
    StdArc::StateId state_threshold = kNoStateId;
    std::vector<StdArc::Weight> distance;

    // create arc filter 

    AnyArcFilter<StdArc> arc_filter;
    //LookAheadComposeFilter<StdArc> arc_filter;

    //LAM* lam1 = new LAM(A, MATCH_OUTPUT);
    //LAM* lam2 = new LAM(B, MATCH_INPUT);
    //
    //LCF* laf1 = new LCF( A, B, lam1, lam2);


    // create queue

    //AutoQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
    //NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>* state_queue =
    //    new NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>(distance);
    NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(distance);

    // create shortest path options

    //const ShortestPathOptions<StdArc, AutoQueue<StdArc::StateId>, AnyArcFilter<StdArc>> opts(
    const ShortestPathOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc>>
        opts(&state_queue, arc_filter, nshortest, unique, false, kDelta, first_path, weight_threshold, state_threshold);


    if (lazy) {

        ComposeFst<StdArc> composed = lazy_compose(input1, input2, input3, word);

        //float compose_time = get_cpu_time();
        //cout << "Composition: " << compose_time << endl;


        //ShortestPath(composed, &nbest_transducer, nbest);

        ShortestPath(composed, &nbest_transducer, &distance, opts);

        //float total_time = get_cpu_time();
        //float search_time = total_time - compose_time;
        //cout << "Search: " << search_time << endl;
        //cout << "Total time: " << total_time << endl;

    }
    else {

        SVF composed = eager_compose(input1, input2, input3, word);

        ////float compose_time = get_cpu_time();
        //cout << "Composition: " << compose_time << endl;

        //ShortestPath(composed, &nbest_transducer, nbest);

        // perform output projection, since unique option in shortest path
        // search requires automaton/acceptor as input
        if (unique) {
            Project(&composed, PROJECT_OUTPUT);
            RmEpsilon(&composed);
        }

        ShortestPath(composed, &nbest_transducer, &distance, opts);

        //float total_time = get_cpu_time();
        //float search_time = total_time - compose_time;
        //cout << "Search: " << search_time << endl;
        //cout << "Total time: " << total_time << endl;

    }

    //if (getrusage(RUSAGE_SELF, &usage) < 0) {
    //    std::perror("cannot get usage statistics");
    //    // exit(1);
    //} else {
    //
    //    // maximum resident set size in kB
    //    std::cout << "RAM usage: " << usage.ru_maxrss << endl;
    //
    //}

    return nbest_transducer;
}


ComposeFst<StdArc> Composition::lazy_compose(
    SVF *input1,
    SVF *input2,
    SVF *input3,
    string word) {

    const SymbolTable table = *(input1->OutputSymbols());
    SVF input_transducer = create_input_transducer(word, &table);
    ArcSort(&input_transducer, StdOLabelCompare());

    //input_transducer.Write(string("input_") + word + string(".fst"));

    //ArcSort(input1, StdOLabelCompare());
    //ArcSort(input2, StdILabelCompare());


    //COpts* opts = new COpts();
    COpts opts;

    //ComposeFst<StdArc>* delayed_result = new ComposeFst<StdArc>(*input1, *input2, opts);
    ComposeFst<StdArc> delayed_result(*input1, *input2, opts);
    ComposeFst<StdArc>* delayed_result2;

    if (input3) {
        //ArcSort(input3, StdILabelCompare());
        //delayed_result2 = new ComposeFst<StdArc>(delayed_result, *input3, opts);
        ComposeFst<StdArc> delayed_result2(delayed_result, *input3, opts);
    }
    
    //delayed_result2 = new ComposeFst<StdArc>(input_transducer, (input3 ? *delayed_result2 : delayed_result), opts);
    ComposeFst<StdArc> delayed_result3(input_transducer, (input3 ? *delayed_result2 : delayed_result), opts);


    return delayed_result3;
}


SVF Composition::eager_compose(
    SVF *input1,
    SVF *input2,
    SVF *input3,
    SVF *input_transducer) {

    const SymbolTable table = *(input1->OutputSymbols());
    //SVF input_transducer = create_input_transducer(word, &table);
    ArcSort(input_transducer, StdOLabelCompare());

    //input_transducer.Write(string("input_") + word + string(".fst"));

    SVF result;

    Compose(*input_transducer, *input1, &result);
 

    if (input2) {
        //ArcSort(&result, StdOLabelCompare());
        Compose(result, *input2, &result);
    }

    if (input3) {
        //ArcSort(&result, StdOLabelCompare());
        Compose(result, *input3, &result);
    }
    
    return result;
}



SVF Composition::eager_compose(
    SVF *input1,
    SVF *input2,
    SVF *input3,
    string word) {

    const SymbolTable table = *(input1->OutputSymbols());
    SVF input_transducer = create_input_transducer(word, &table);
    ArcSort(&input_transducer, StdOLabelCompare());

    //input_transducer.Write(string("input_") + word + string(".fst"));

    SVF result;

    Compose(input_transducer, *input1, &result);

    if (input2) {
        //ArcSort(&result, StdOLabelCompare());
        Compose(result, *input2, &result);
    }

    if (input3) {
        //ArcSort(&result, StdOLabelCompare());
        Compose(result, *input3, &result);
    }
    
    return result;
}


SVF Composition::create_input_transducer(string word, const SymbolTable* table) {

    SVF input;
    input.SetInputSymbols(table);
    input.SetOutputSymbols(table);
    input.AddState();
    input.SetStart(0);


    wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    //string narrow = converter.to_bytes(wide_utf16_source_string);
    wstring wide = converter.from_bytes(word);
    //cout << "Word: " << converter.to_bytes(wide) << endl;

    for (int i=0; i < wide.length(); i++) {

        //cout << converter.to_bytes(wide[i]) << endl;
        //char letter =  word[i];
        string narrow_symbol = converter.to_bytes(wide[i]);

        int label = table->Find(narrow_symbol);
        //cout << label << endl;
        input.AddArc(i, StdArc(label, label, 0, i+1));
        input.AddState();

    }
    input.SetFinal(wide.length(), 0);
    return input;
}



