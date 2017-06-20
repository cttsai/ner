#include "Utils.h"

vector<string> *types = new vector<string>;
unordered_map<string, string> *label2id = new unordered_map<string, string>;
unordered_map<int, string> *id2label = new unordered_map<int, string>;

void pronominal_exp(){
    types->push_back("LOC");
    types->push_back("ORG");
    types->push_back("PER");
    types->push_back("GPE");
    types->push_back("FAC");

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

    const char *train_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1/";
//    const char *train_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1.train/";
//    const char *test_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1.test/";

    const char *test_dir = "/shared/experiments/ctsai12/workspace/xlwikifier-demo/TAC2016.df2.spanish.predictions/";

    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->filter_features = true;
//    extractor->read_good_features("gf_nominal_1");
//    extractor->use_brown = false;
    extractor->use_wikifier = false;
    extractor->use_hyphen = false;
    extractor->use_gazetteer = false;
    extractor->use_tagcontext = false;
    extractor->use_tagpattern = false;
    extractor->use_pretag2 = false;
    extractor->use_pretag1 = false;
    extractor->use_wordtype = false;
    extractor->use_affixe = false;
    extractor->use_cap = false;
//    extractor->use_sen = false;
    extractor->form_context_size = 1;
    extractor->brown_context_size = 1;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 0.125;
//    double c = 4;

    vector<Document *> *test_docs;

    double max_f1 = 0;
    double max_seg = 0;
    for(int i = 0; i < 1; i++){

        cout << "c " << c  << endl;
        struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, 1, 1);
        cout << "#features " << model->nr_feature << endl;

        test_docs = readColumnFormatFiles(test_dir);
        clean_cap_words(test_docs);
        predict_ner(test_docs, extractor, model);
        free_and_destroy_model(&model);
        bilou_to_bio(test_docs);
        double f1 = evaluate_phrases(test_docs);
        writeTACFormat(test_docs, "df2.pro", "PRO");
        max_f1 = max(max_f1, f1);
        free_docs(test_docs);
        c/=2;
    }
    cout << "max f1 " << max_f1 << endl;
}

void nominal_exp(){
    types->push_back("LOC");
    types->push_back("ORG");
    types->push_back("PER");
    types->push_back("GPE");
    types->push_back("FAC");

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

//    const char *train_dir = "/shared/corpora/ner/nominal_exp/es.tac.NOM.train-nerf/";
    const char *train_dir = "/shared/corpora/ner/nominal_exp/es.ere.NOM.spe/";
    const char *test_dir = "/shared/corpora/ner/nominal_exp/es.tac.NOM-nerf/";

//    test_dir = "/shared/experiments/ctsai12/workspace/xlwikifier-demo/TAC2016.df2.spanish.predictions/";

    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->filter_features = true;
//    extractor->read_good_features("gf_nominal_1");
//    extractor->use_brown = false;
    extractor->use_wikifier = false;
    extractor->use_hyphen = false;
    extractor->use_gazetteer = false;
    extractor->use_tagcontext = false;
    extractor->use_tagpattern = false;
    extractor->use_pretag2 = false;
    extractor->use_pretag1 = false;
    extractor->use_wordtype = false;
    extractor->use_affixe = false;
    extractor->use_cap = false;
    extractor->use_sen = false;
    extractor->form_context_size = 0;
    extractor->brown_context_size = 0;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

//    double c = 0.0625;
    double c = 4;

    vector<Document *> *test_docs;

    double max_f1 = 0;
    double max_seg = 0;
    for(int i = 0; i < 10; i++){

        cout << "c " << c << endl;
        struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, 1, 1);
        cout << "#features " << model->nr_feature << endl;
        test_docs = readColumnFormatFiles(test_dir);
        clean_cap_words(test_docs);
        predict_ner(test_docs, extractor, model);
        free_and_destroy_model(&model);
        bilou_to_bio(test_docs);
        double f1 = evaluate_phrases(test_docs);
//                writeTACFormat(test_docs, "df2");
        max_f1 = max(max_f1, f1);
        free_docs(test_docs);
        c/=2;
    }
    cout << "max f1 " << max_f1 << endl;
}

void conll_exp(){
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

    const char *train_dir = "/shared/corpora/ner/wikifier-features/en/train-camera3/";
    const char *test_dir = "/shared/corpora/ner/wikifier-features/en/test-camera3/";

//    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/TrainPlusDev/";
//    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
//    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

//    train_dir = "/shared/corpora/ner/wikifier-features/tl/train-camera3/";
//    test_dir = "/shared/corpora/ner/wikifier-features/tl/test-camera3/";


    FeatureExtractor *extractor = new FeatureExtractor();
    extractor->read_good_features("good_features_0.15_fw3");
//    extractor->read_good_features1("good_features_0.125");
//    extractor->form_context_size = 3;
//    extractor->brown_context_size = 3;
//    extractor->gazetteer_context_size = 1;
//    extractor->use_gazetteer = false;
//    extractor->use_brown = false;
//    extractor->use_affixe = false;
//    extractor->use_tagpattern = false;
//    extractor->use_tagcontext = false;
//    extractor->use_form = false;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);
//    struct problem **train_probs = build_svm_problem1(train_docs, extractor);

    extractor->training = false;

    double c = 0.5;

//    select_features(train_prob, extractor, c, "good_features_0.17_fw3");
//    select_features1(train_prob, orig_label, extractor, c, "good_features_0.125");

    vector<Document *> *test_docs;
    vector<Document *> *dev_docs;
    ofstream paramfile("tmp");

    for(int i = 0; i < 1; i++){
        double gamma = 1;
        for(int j = 0; j < 1; j++) {
            double coef = 8;
            for (int k = 0; k < 1; k++) {

                cout << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);
//                struct model **models = train_ner1(train_probs, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);
//                cout << "#features " << model->nr_feature << endl;

                // dev
                dev_docs = readColumnFormatFiles(dev_dir);
                clean_cap_words(dev_docs);
                predict_ner(dev_docs, extractor, model);
//                predict_ner1(dev_docs, extractor, models);
                bilou_to_bio(dev_docs);
                double f1_dev = evaluate_phrases(dev_docs);
                free_docs(dev_docs);

                // test
                test_docs = readColumnFormatFiles(test_dir);
                clean_cap_words(test_docs);
                predict_ner(test_docs, extractor, model);
                free_and_destroy_model(&model);
//                predict_ner1(test_docs, extractor, models);
//                free_models(models);
                bilou_to_bio(test_docs);
                double f1 = evaluate_phrases(test_docs);
                paramfile << f1_dev << " " << f1 << endl;
//                coef /= 2;
                coef -= 1;
                free_docs(test_docs);
            }
//            gamma -= 0.25;
            gamma /= 2;
        }
        c/=2;
//        c -= 0.1;
    }
//    paramfile << "max f1 " << max_f1 << endl;
    paramfile.close();
}

void semisup_exp(){
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

    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

//    const char *unlabel_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Ontonotes/ColumnFormat/Train/";
    const char *unlabel_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";

    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->use_gazetteer = false;
//    extractor->use_brown = false;


    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    vector<Document *> *unlabel_docs = readColumnFormatFiles(unlabel_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 4;


    vector<Document *> *test_docs;
    vector<Document *> *dev_docs;
//    ofstream paramfile("ps_tmp");

    for(int i = 0; i < 1; i++){

        cout << "c " << c  << endl;
//        paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
        struct model *model = train_ner(train_prob, L2R_LR_DUAL, c, 0.1, 1, 1);
        cout << "#features " << model->nr_feature << endl;
        
        struct problem *semi_prob = build_semi_problem(unlabel_docs, extractor, model);
        struct model *model1 = train_ner(semi_prob, L2R_L2LOSS_SVC, c, 0.1, 1, 1);

        // dev
        dev_docs = readColumnFormatFiles(dev_dir);
        clean_cap_words(dev_docs);
        predict_ner(dev_docs, extractor, model1);
        bilou_to_bio(dev_docs);
        double f1_dev = evaluate_phrases(dev_docs);
        free_docs(dev_docs);

        // test
        test_docs = readColumnFormatFiles(test_dir);
        clean_cap_words(test_docs);
        predict_ner(test_docs, extractor, model1);
        free_and_destroy_model(&model);
        bilou_to_bio(test_docs);
        double f1 = evaluate_phrases(test_docs);
        free_docs(test_docs);
        c/=2;
//        paramfile << f1_dev << " " << f1 << endl;
    }
//    paramfile << "max f1 " << max_f1 << endl;
//    paramfile.close();

}

int main() {

//    semisup_exp();
    conll_exp();
//    nominal_exp();
//    pronominal_exp();

    return 0;
}