#include "Utils.h"

vector<string> *types = new vector<string>;
unordered_map<string, string> *label2id = new unordered_map<string, string>;
unordered_map<int, string> *id2label = new unordered_map<int, string>;

void set_conll_types(){
    types->push_back("LOC");
    types->push_back("ORG");
    types->push_back("PER");
    types->push_back("MISC");

    int cnt = 0;
    vector<string>::iterator it;
    for(it = types->begin(); it != types->end(); ++it){
        string type = *it;
        label2id->insert({"B-"+type, to_string(cnt)});
        id2label->insert({cnt++, "B-"+type});
        label2id->insert({"I-"+type, to_string(cnt)});
        id2label->insert({cnt++, "I-"+type});
        label2id->insert({"L-"+type, to_string(cnt)});
        id2label->insert({cnt++, "L-"+type});
        label2id->insert({"U-"+type, to_string(cnt)});
        id2label->insert({cnt++, "U-"+type});
    }
    label2id->insert({"O", to_string(cnt)});
    id2label->insert({cnt, "O"});
}


void conll_exp(){

    set_conll_types();

//    const char *train_dir = "/shared/corpora/ner/wikifier-features/en/train-camera3/";
//    const char *test_dir = "/shared/corpora/ner/wikifier-features/en/test-camera3/";

    //const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/TrainPlusDev/";
    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";


    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->read_good_features("good_features_0.15_wikibc1");
//    extractor->form_context_size = 2;
//    extractor->brown_context_size = 3;
//    extractor->gazetteer_context_size = 1;
//    extractor->use_gazetteer = false;
//    extractor->use_brown = false;
//    extractor->form_conj = false;
    extractor->gazetteer_list = "/home/ctsai12/CLionProjects/NER/gazetteers-list.txt__";
    extractor->form_context_size = 3;
    //extractor->form_conj = false;
    extractor->prefix_len.push_back(20);

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 0.125;

    //select_features(train_prob, extractor, c, "good_features_rep1");

    vector<Document *> *test_docs;
    vector<Document *> *dev_docs;
    ofstream paramfile("tmp");

    for(int i = 0; i < 10; i++){
        double gamma = 0.5;
        for(int j = 0; j < 1; j++) {
            double coef = 16;
            for (int k = 0; k < 1; k++) {

                cout << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                struct model *model = train_ner(train_prob, MCSVM_CS, c, 0.1, gamma, coef);

                // dev
                dev_docs = readColumnFormatFiles(dev_dir);
                clean_cap_words(dev_docs);
                predict_ner(dev_docs, extractor, model);
                bilou_to_bio(dev_docs);
                double f1_dev = evaluate_phrases(dev_docs);
                free_docs(dev_docs);

                // test
                test_docs = readColumnFormatFiles(test_dir);
                clean_cap_words(test_docs);
                predict_ner(test_docs, extractor, model);
                free_and_destroy_model(&model);
                bilou_to_bio(test_docs);
                double f1 = evaluate_phrases(test_docs);
                paramfile << f1_dev << " " << f1 << endl;
                coef /= 2;
                //coef -= 1;
                free_docs(test_docs);
            }
            gamma += 0.5;
//            gamma /= 2;
        }
        c/=2;
//        c -= 0.1;
    }
    paramfile.close();
}

void run_sota(){

#ifndef POLY2
    cout << "Use poly2 version of liblinear to get the best performance" << endl;
    cout << "Use linear-poly2.h in Utils.h and linear-poly2.o in CMakeLists.txt" << endl;
    exit(-1);
#endif

    set_conll_types();

    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

    FeatureExtractor *extractor = new FeatureExtractor();
    //extractor->read_good_features("/home/ctsai12/CLionProjects/NER/resources/good_features_0.15_b20");
    extractor->read_good_features("/home/ctsai12/CLionProjects/NER/good_features_rep");
    extractor->form_context_size = 3;
    extractor->form_conj = false;
    extractor->prefix_len.push_back(20);

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    double c = 0.5, gamma = 0.5, coef = 4;

    struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);

    // dev
    vector<Document *> *dev_docs = readColumnFormatFiles(dev_dir);
    clean_cap_words(dev_docs);
    predict_ner(dev_docs, extractor, model);
    bilou_to_bio(dev_docs);
    evaluate_phrases(dev_docs);
    free_docs(dev_docs);

    // test
    vector<Document *> *test_docs = readColumnFormatFiles(test_dir);
    clean_cap_words(test_docs);
    predict_ner(test_docs, extractor, model);
    free_and_destroy_model(&model);
    bilou_to_bio(test_docs);
    evaluate_phrases(test_docs);
    free_docs(test_docs);
}



int main() {

    conll_exp();
//    run_sota();

    return 0;
}
