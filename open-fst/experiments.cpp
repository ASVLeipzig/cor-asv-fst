#include <fst/fstlib.h>

#include <stdio.h>
#include <iterator>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <pthread.h>
#include <condition_variable>

#include <locale>
#include <codecvt>
#include <string>

#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>

//#include <cunistd>
#include <sys/time.h>
#include <sys/resource.h>
#include <ctime>


//g++ -std=c++14 -I /usr/local/include -ldl -lfst -lpthread experiments.cpp -o experiments
//g++ -std=c++14 -I /usr/local/include -ldl -lfst -lpthread -gp experiments.cpp -o experiments


using namespace fst;
using namespace std;


pid_t child_pid;
int status;

struct rusage usage;


void kill_child(int sig) {

    kill(child_pid, SIGTERM);
}


StdVectorFst create_input_transducer(string word, const SymbolTable* table) {

    StdVectorFst input;
    input.SetInputSymbols(table);
    input.SetOutputSymbols(table);
    input.AddState();
    input.SetStart(0);


    wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    //string narrow = converter.to_bytes(wide_utf16_source_string);
    wstring wide = converter.from_bytes(word);
    cout << "Word: " << converter.to_bytes(wide) << endl;

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


ComposeFst<StdArc> lazy_compose(
    StdVectorFst *input1,
    StdVectorFst *input2,
    StdVectorFst *input3,
    string word) {

    const SymbolTable table = *(input1->OutputSymbols());
    StdVectorFst input_transducer = create_input_transducer(word, &table);
    ArcSort(&input_transducer, StdOLabelCompare());

    //input_transducer.Write(string("input_") + word + string(".fst"));

    //ArcSort(input1, StdOLabelCompare());
    //ArcSort(input2, StdILabelCompare());


    typedef StdArc A;
    typedef A::StateId S;
    typedef A::Weight W;
    typedef Matcher< Fst<A> > M;
    typedef SequenceComposeFilter<M> F;
    typedef GenericComposeStateTable<A, F::FilterState> T;
    typedef ComposeFstOptions<A, M, F, T> COpts;

    COpts* opts = new COpts();

    ComposeFst<StdArc>* delayed_result = new ComposeFst<StdArc>(*input1, *input2, *opts);
    ComposeFst<StdArc>* delayed_result2;

    if (input3) {
        //ArcSort(input3, StdILabelCompare());
        delayed_result2 = new ComposeFst<StdArc>(*delayed_result, *input3, *opts);
    }
    
    delayed_result2 = new ComposeFst<StdArc>(input_transducer, (input3 ? *delayed_result2 : *delayed_result), *opts);

    return *delayed_result2;
}


StdVectorFst eager_compose(
    StdVectorFst *input1,
    StdVectorFst *input2,
    StdVectorFst *input3,
    string word) {

    const SymbolTable table = *(input1->OutputSymbols());
    StdVectorFst input_transducer = create_input_transducer(word, &table);
    ArcSort(&input_transducer, StdOLabelCompare());

    //input_transducer.Write(string("input_") + word + string(".fst"));

    StdVectorFst result;

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


float get_cpu_time() {

    if (getrusage(RUSAGE_SELF, &usage) < 0) {
        std::perror("cannot get usage statistics");
        // exit(1);
        return -1;
    } else {
    
        return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
            1e-6*usage.ru_utime.tv_usec + 1e-6*usage.ru_stime.tv_usec;
    
    }
}


StdVectorFst compose_and_search(
    StdVectorFst *input1,
    StdVectorFst *input2,
    StdVectorFst *input3,
    string word,
    bool lazy,
    int nbest) {

    StdVectorFst nbest_transducer;


    if (lazy) {

        ComposeFst<StdArc> composed = lazy_compose(input1, input2, input3, word);

        float compose_time = get_cpu_time();
        cout << "Composition: " << compose_time << endl;

        //int32 nshortest = 1;
        int32 nshortest = nbest;
        bool unique = false;
        bool first_path = false;
        StdArc::Weight weight_threshold = StdArc::Weight::Zero();
        StdArc::StateId state_threshold = kNoStateId;

        std::vector<StdArc::Weight> distance;
        AnyArcFilter<StdArc> arc_filter;
        AutoQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
        //ShortestFirstQueue<StdArc::StateId> state_queue(composed, &distance, arc_filter);
        const ShortestPathOptions<StdArc, AutoQueue<StdArc::StateId>, AnyArcFilter<StdArc>> opts(
        &state_queue, arc_filter, nshortest, unique, false, kDelta, first_path,
        weight_threshold, state_threshold);


        //ShortestPath(composed, &nbest_transducer, nbest);

        ShortestPath(composed, &nbest_transducer, &distance, opts);

        float total_time = get_cpu_time();
        float search_time = total_time - compose_time;
        cout << "Search: " << search_time << endl;
        cout << "Total time: " << total_time << endl;

    }
    else {

        StdVectorFst composed = eager_compose(input1, input2, input3, word);

        float compose_time = get_cpu_time();
        cout << "Composition: " << compose_time << endl;

        ShortestPath(composed, &nbest_transducer, nbest);

        float total_time = get_cpu_time();
        float search_time = total_time - compose_time;
        cout << "Search: " << search_time << endl;
        cout << "Total time: " << total_time << endl;

    }

    if (getrusage(RUSAGE_SELF, &usage) < 0) {
        std::perror("cannot get usage statistics");
        // exit(1);
    } else {
    
        // maximum resident set size in kB
        std::cout << "RAM usage: " << usage.ru_maxrss << endl;
    
    }

    return nbest_transducer;
}


//StdVectorFst compose_and_search(
//    StdVectorFst *input1,
//    StdVectorFst *input2,
//    StdVectorFst *input3,
//    string word,
//    bool lazy,
//    int nbest) {
//
//    StdVectorFst nbest_transducer;
//
//    if (lazy) {
//
//        ComposeFst<StdArc> composed = lazy_compose(input1, input2, input3, word);
//
//        float compose_time = get_cpu_time();
//        cout << "Composition: " << compose_time << endl;
//
//        ShortestPath(composed, &nbest_transducer, nbest);
//
//        float total_time = get_cpu_time();
//        float search_time = total_time - compose_time;
//        cout << "Search: " << search_time << endl;
//        cout << "Total time: " << total_time << endl;
//
//    }
//    else {
//
//        StdVectorFst composed = eager_compose(input1, input2, input3, word);
//
//        float compose_time = get_cpu_time();
//        cout << "Composition: " << compose_time << endl;
//
//        ShortestPath(composed, &nbest_transducer, nbest);
//
//        float total_time = get_cpu_time();
//        float search_time = total_time - compose_time;
//        cout << "Search: " << search_time << endl;
//        cout << "Total time: " << total_time << endl;
//
//    }
//
//    if (getrusage(RUSAGE_SELF, &usage) < 0) {
//        std::perror("cannot get usage statistics");
//        // exit(1);
//    } else {
//    
//        // maximum resident set size in kB
//        std::cout << "RAM usage: " << usage.ru_maxrss << endl;
//    
//    }
//
//    return nbest_transducer;
//}


StdVectorFst composition_wrapper(
    StdVectorFst *input1,
    StdVectorFst *input2,
    StdVectorFst *input3,
    string word,
    bool lazy,
    int nbest) {

    mutex m;
    condition_variable cv;
    StdVectorFst return_value;

    thread t([&m, &cv, &return_value, input1, input2, input3, word, lazy, nbest]() {

        return_value = compose_and_search(input1, input2, input3, word, lazy, nbest);
        cv.notify_one();
    });

    t.detach();

    {
        unique_lock<mutex> l(m);
        if(cv.wait_for(l, 10s) == cv_status::timeout) 
            throw runtime_error("Timeout");
    }

    return return_value;    
}


StdVectorFst perform_experiment(
    StdVectorFst *input1,
    StdVectorFst *input2,
    StdVectorFst *input3,
    string word,
    bool lazy,
    int nbest) {

    StdVectorFst nbest_transducer;

    bool timedout = false;
    try {
        nbest_transducer = composition_wrapper(input1, input2, input3, word, lazy, nbest);
    }
    catch(std::runtime_error& e) {
        cout << e.what() << endl;
        timedout = true;
    }

    if(!timedout)
        cout << "Success" << endl;

    return nbest_transducer;

}


int main(int argc, const char* argv[]) {

    //chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    // read transducer files

    StdVectorFst *error_model1 = StdVectorFst::Read("error_transducers/max_error_1.ofst");
    StdVectorFst *error_model2 = StdVectorFst::Read("error_transducers/max_error_2.ofst");
    StdVectorFst *error_model3 = StdVectorFst::Read("error_transducers/max_error_3.ofst");
    StdVectorFst *error_model4 = StdVectorFst::Read("error_transducers/max_error_4.ofst");
    StdVectorFst *error_model5 = StdVectorFst::Read("error_transducers/max_error_5.ofst");

    StdVectorFst *context_error_model1 = StdVectorFst::Read("context/max_error_1_context_2_3.ofst");
    StdVectorFst *context_error_model2 = StdVectorFst::Read("context/max_error_2_context_2_3.ofst");
    StdVectorFst *context_error_model3 = StdVectorFst::Read("context/max_error_3_context_2_3.ofst");
    StdVectorFst *context_error_model4 = StdVectorFst::Read("context/max_error_4_context_2_3.ofst");
    StdVectorFst *context_error_model5 = StdVectorFst::Read("context/max_error_5_context_2_3.ofst");

    StdVectorFst *context_error_model_1_1 = StdVectorFst::Read("context/max_error_1_context_1.ofst");
    StdVectorFst *context_error_model_1_2 = StdVectorFst::Read("context/max_error_2_context_1.ofst");
    StdVectorFst *context_error_model_1_3 = StdVectorFst::Read("context/max_error_3_context_1.ofst");
    StdVectorFst *context_error_model_1_4 = StdVectorFst::Read("context/max_error_4_context_1.ofst");
    StdVectorFst *context_error_model_1_5 = StdVectorFst::Read("context/max_error_5_context_1.ofst");

    StdVectorFst *context_error_model_2_1 = StdVectorFst::Read("context/max_error_1_context_2.ofst");
    StdVectorFst *context_error_model_2_2 = StdVectorFst::Read("context/max_error_2_context_2.ofst");
    StdVectorFst *context_error_model_2_3 = StdVectorFst::Read("context/max_error_3_context_2.ofst");
    StdVectorFst *context_error_model_2_4 = StdVectorFst::Read("context/max_error_4_context_2.ofst");
    StdVectorFst *context_error_model_2_5 = StdVectorFst::Read("context/max_error_5_context_2.ofst");

    StdVectorFst *context_error_model_3_1 = StdVectorFst::Read("context/max_error_1_context_3.ofst");
    StdVectorFst *context_error_model_3_2 = StdVectorFst::Read("context/max_error_2_context_3.ofst");
    StdVectorFst *context_error_model_3_3 = StdVectorFst::Read("context/max_error_3_context_3.ofst");
    StdVectorFst *context_error_model_3_4 = StdVectorFst::Read("context/max_error_4_context_3.ofst");
    StdVectorFst *context_error_model_3_5 = StdVectorFst::Read("context/max_error_5_context_3.ofst");

    StdVectorFst *lexicon_small = StdVectorFst::Read("lexicon_transducers/lexicon.ofst");
    StdVectorFst *lexicon_big = StdVectorFst::Read("lexicon_transducers/lexicon_transducer_asse_minimized.ofst");

    StdVectorFst *extended_lexicon_small = StdVectorFst::Read("extended_lexicon/extended_lexicon.ofst");

    StdVectorFst *rules = StdVectorFst::Read("morphology/rules.ofst");
    //StdVectorFst *rules = StdVectorFst::Read("morphology/optionalized_rules.ofst");

    // list all transducers whose arcs need to be sorted

    vector<StdVectorFst*> arc_sort_required = {
        error_model1,
        error_model2,
        error_model3,
        error_model4,
        error_model5,
        context_error_model1,
        context_error_model2,
        context_error_model3,
        context_error_model4,
        context_error_model5,
        context_error_model_1_1,
        context_error_model_1_2,
        context_error_model_1_3,
        context_error_model_1_4,
        context_error_model_1_5,
        context_error_model_2_1,
        context_error_model_2_2,
        context_error_model_2_3,
        context_error_model_2_4,
        context_error_model_2_5,
        context_error_model_3_1,
        context_error_model_3_2,
        context_error_model_3_3,
        context_error_model_3_4,
        context_error_model_3_5,
        lexicon_small,
        lexicon_big,
        extended_lexicon_small,
        rules
    };


    // error models

    vector<StdVectorFst*> error_models_123 = {error_model1, error_model2, error_model3, error_model4, error_model5};

    vector<StdVectorFst*> context_error_models_1 = {context_error_model_1_1, context_error_model_1_2,
        context_error_model_1_3, context_error_model_1_4, context_error_model_1_5};

    vector<StdVectorFst*> context_error_models_2 = {context_error_model_2_1, context_error_model_2_2,
        context_error_model_2_3, context_error_model_2_4, context_error_model_2_5};

    vector<StdVectorFst*> context_error_models_3 = {context_error_model_3_1, context_error_model_3_2,
        context_error_model_3_3, context_error_model_3_4, context_error_model_3_5};

    vector<StdVectorFst*> context_error_models_23 = {context_error_model1, context_error_model2,
        context_error_model3, context_error_model4, context_error_model5};

    vector<vector<StdVectorFst*>> all_error_models = {error_models_123, context_error_models_1,
        context_error_models_2, context_error_models_3, context_error_models_23};


    //chrono::steady_clock::time_point end = chrono::steady_clock::now();
    //cout << "Load Transducers: " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << endl;


    // parameters for experiments

    int num_errors = 1;
    //StdVectorFst *error_model = error_model3;

    //bool use_morphology = true;

    bool big_lexicon = true;

    bool extended_lexicon = false; // lexicon + morphology, only for use of morphology
    bool precomposed = false; // error + lexicon, not in combination with extended_lexicon
    int nbest = 1;
    bool lazy = false;
    int context = 3;

    //string input_words[] = {"bIeibt", "zuständigen", "miüssen", "radioalctiver",
    //    "schiefßen", "niedersBchsischen"};



    // select lexicon

    //if (big_lexicon) {
    //    lexicon = lexicon_big;
    //}
    //else {
    //    lexicon = lexicon_small;
    //}

    //if (extended_lexicon and !big_lexicon) {
    //    lexicon = extended_lexicon_small;
    //}

    //StdVectorFst* lexicon = lexicon_big; // set for SymbolTable merge
    
    // adjust output and input symbol tables for composition

    bool relabel;

    const SymbolTable output_symbols = *(all_error_models[0][0]->OutputSymbols());
    const SymbolTable input_symbols = *(lexicon_big->InputSymbols());

    SymbolTable lexicon_new = *(MergeSymbolTable(output_symbols, input_symbols, &relabel));

    const SymbolTable small_lexicon_symbols = *(lexicon_small->InputSymbols());

    lexicon_new = *(MergeSymbolTable(lexicon_new, small_lexicon_symbols, NULL));

    const SymbolTable rules_symbols = *(rules->InputSymbols());

    lexicon_new = *(MergeSymbolTable(lexicon_new, rules_symbols, NULL));
    //const combined_lexicon_new = *(MergeSymbolTable(lexicon_new, rules_symbols, NULL));

    if (relabel) {
      Relabel(lexicon_big, &lexicon_new, &lexicon_new);
      Relabel(lexicon_small, &lexicon_new, &lexicon_new);
      //Relabel(error_model, &lexicon_new, nullptr);
      Relabel(rules, &lexicon_new, &lexicon_new);
    }

    lexicon_big->SetOutputSymbols(&lexicon_new);
    lexicon_big->SetInputSymbols(&lexicon_new);

    lexicon_small->SetOutputSymbols(&lexicon_new);
    lexicon_small->SetInputSymbols(&lexicon_new);

    rules->SetOutputSymbols(&lexicon_new);
    rules->SetInputSymbols(&lexicon_new);

    
    for (int h = 0; h < std::end(all_error_models) - std::begin(all_error_models); h++) {
        for (int i = 0; i < std::end(all_error_models[h]) - std::begin(all_error_models[h]); i++) {
            all_error_models[h][i]->SetOutputSymbols(&lexicon_new);
            all_error_models[h][i]->SetInputSymbols(&lexicon_new);
        }
    }

    
    // perfom arc sort

    for (int i = 0; i < std::end(arc_sort_required) - std::begin(arc_sort_required); i++) {

        ArcSort(arc_sort_required[i], StdOLabelCompare());
        ArcSort(arc_sort_required[i], StdILabelCompare());

    }

    // use morphology
    //if (!use_morphology) {
    //    rules = NULL;
    //}


    // compose and search

    StdVectorFst nbest_transducer;

    vector<string> input_words = {"bIeibt", "zuständigen", "miüssen", "radioalctiver",
        "schiefßen", "niedersBchsischen"};

    vector<bool> lazy_options = {true};
    vector<StdVectorFst*> lexicons = {lexicon_small, lexicon_big};
    vector<StdVectorFst*> morphologies = {NULL, rules};
    //vector<StdVectorFst*> morphologies = {rules};
    vector<int> error_model_options = {123, 1, 2, 3, 23};

    vector<int> error_numbers = {1, 2, 3, 4, 5};
    vector<int> nbests = {1, 10, 50};


    //vector<string> input_words = {"bIeibt", "zuständigen"};

    //vector<bool> lazy_options = {false};
    //vector<StdVectorFst*> lexicons = {lexicon_big};
    //vector<StdVectorFst*> morphologies = {rules};
    //vector<int> error_model_options = {123};

    //vector<int> error_numbers = {1, 2, 3, 4, 5};
    //vector<int> nbests = {1};


    for (int d = 0; d < std::end(lazy_options) - std::begin(lazy_options); d++) {
        lazy = lazy_options[d];

        for (int e = 0; e < std::end(lexicons) - std::begin(lexicons); e++) {
            StdVectorFst* lexicon = lexicons[e];

            for (int f = 0; f < std::end(morphologies) - std::begin(morphologies); f++) {
                StdVectorFst* morphology = morphologies[f];

                for (int g = 0; g < std::end(error_model_options) - std::begin(error_model_options); g++) {
                    context = error_model_options[g];
                    vector<StdVectorFst*> error_models;
                    

                    switch (context) {
                        case 23 :
                            error_models = context_error_models_23;
                            break;
                        case 1:
                            error_models = context_error_models_1;
                            break;
                        case 2 :
                            error_models = context_error_models_2;
                            break;
                        case 3 :
                            error_models = context_error_models_3;
                            break;
                        default : 
                            error_models = error_models_123;
                            break;
                    }

                    for (int h = 0; h < std::end(error_numbers) - std::begin(error_numbers); h++) {

                        for (int j = 0; j < std::end(nbests) - std::begin(nbests); j++) {

                            for (int i = 0; i < std::end(input_words) - std::begin(input_words); i++) {
                                string word = input_words[i];

                                cout << "Context: " << context << endl;
                                cout << "Number of errors: " << error_numbers[h] << endl;
                                cout << "n-best: " << nbests[j] << endl;
                                cout << "Lazy: " << lazy << endl;
                                cout << "Morphology: " << f << endl;
                                cout << "Lexicon: " << e << endl;

                                if ((child_pid = fork()) < 0) {
                                    std::perror("fork() failed");
                                }
                                else if (child_pid == 0) {
                                    alarm(10); // give up after 10s

                                    nbest_transducer = compose_and_search(
                                        error_models[error_numbers[h] - 1],
                                        lexicon,
                                        morphology,
                                        word,
                                        lazy,
                                        nbests[j]);

                                    nbest_transducer.Write(word + string(".fst"));
                                    cout << "States: " << nbest_transducer.NumStates() << endl;
                                    exit(0);
                                 }
                                 else {
                                    if (wait(&status) < 0) {
                                        std::perror("wait() failed");
                                    }
                                    else {
                                        if (WIFEXITED(status)) 
                                            //std::cout << "child " << i << " exited with " << WEXITSTATUS(status) << endl;
                                            std::cout << "Timeout: 0" << endl;
                                
                                        else if (WIFSIGNALED(status))
                                            //std::cout << "child " << i << " aborted after signal " << WTERMSIG(status) << endl;
                                            std::cout << "Timeout: 1" << endl;
                                    }
                                    cout << endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //bool timedout = false;
    //try {
    //    nbest_transducer = composition_wrapper(error_model, lexicon, NULL, word, lazy, 1);
    //}
    //catch(std::runtime_error& e) {
    //    std::cout << e.what() << std::endl;
    //    timedout = true;
    //}

    //if(!timedout)
    //    std::cout << "Success" << std::endl;


    // project output

    //Project(&nbest_transducer, PROJECT_OUTPUT);

    //nbest_transducer.Write("result.fst");


    return 0;

}
