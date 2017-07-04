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
void set_lorelei_types(){
    types->push_back("LOC");
    types->push_back("ORG");
    types->push_back("PER");
    types->push_back("GPE");

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

void set_tac_types(){
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
}

void pronominal_exp(){

    set_tac_types();

    const char *train_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1/";
//    const char *train_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1.train/";
//    const char *test_dir = "/shared/corpora/ner/pronominal_exp/es/es.ere.PRO.spe1.test/";

//    const char *test_dir = "/shared/experiments/ctsai12/workspace/xlwikifier-demo/TAC2016.df2.spanish.predictions/";
    const char *test_dir = "/shared/bronte/Tinkerbell/EDL/cold_start_outputs/es/TAC2017.es.conll/";

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
    extractor->min_word_freq = 3;
    extractor->brown_cluster_paths.clear();
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt");
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brownBllipClusters");
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/es/wiki-c1000-min3");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/en/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/es/wiki-c500-min3");

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
//    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 0.5; // use this for eval
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
        //double f1 = evaluate_phrases(test_docs);
        writeTACFormat(test_docs, "/shared/bronte/Tinkerbell/EDL/cold_start_outputs/es/TAC2017.es.nw.pro", "PRO");
        //max_f1 = max(max_f1, f1);
        free_docs(test_docs);
        c/=2;
    }
    cout << "max f1 " << max_f1 << endl;
}

void nominal_exp(){

    set_tac_types();

    const char *train_dir = "/shared/corpora/ner/nominal_exp/ere+tac/"; // use this for eval!!

//    const char *train_dir = "/shared/corpora/ner/nominal_exp/es.tac.NOM.train-nerf/";
//    const char *train_dir = "/shared/corpora/ner/nominal_exp/es.ere.NOM1.spe/";
//    const char *train_dir = "/shared/corpora/ner/nominal_exp/ere+train/";
//    const char *test_dir = "/shared/corpora/ner/nominal_exp/es.tac.NOM-nerf/";
//    const char *test_dir = "/shared/corpora/ner/nominal_exp/es.tac.NOM.test-nerf/";

    //const char *test_dir = "/shared/experiments/ctsai12/workspace/xlwikifier-demo/TAC2016.df2.spanish.predictions/";
    const char *test_dir = "/shared/bronte/Tinkerbell/EDL/cold_start_outputs/es/TAC2017.es.conll/";

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
    extractor->form_context_size = 1;
    extractor->brown_context_size = 1;
    extractor->min_word_freq = 3;
    extractor->brown_cluster_paths.clear();
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt");
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brownBllipClusters");
    extractor->brown_cluster_paths.push_back("/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/es/wiki-c1000-min3");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/en/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt");
    extractor->brown_cluster_paths.push_back("/shared/preprocessed/ctsai12/multilingual/xlwikifier-data/brown-clusters/es/wiki-c500-min3");

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
//    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

//    double c = 0.0625;

    double c = 0.03125; // use this for eval!

    vector<Document *> *test_docs;

    double max_f1 = 0;
    double max_seg = 0;
    for(int i = 0; i < 1; i++){
        cout << "c " << c << endl;
        struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, 1, 1);
        cout << "#features " << model->nr_feature << endl;
        test_docs = readColumnFormatFiles(test_dir);
        clean_cap_words(test_docs);
        predict_ner(test_docs, extractor, model);
        free_and_destroy_model(&model);
        bilou_to_bio(test_docs);
        //double f1 = evaluate_phrases(test_docs);
        writeTACFormat(test_docs, "/shared/bronte/Tinkerbell/EDL/cold_start_outputs/es/TAC2017.es.nw.nom", "NOM");
        //max_f1 = max(max_f1, f1);
        free_docs(test_docs);
        c/=2;
    }
    cout << "max f1 " << max_f1 << endl;
}

void conll_exp(){

    set_conll_types();

//    const char *train_dir = "/shared/corpora/ner/wikifier-features/en/train-camera3/";
//    const char *test_dir = "/shared/corpora/ner/wikifier-features/en/test-camera3/";

//    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/TrainPlusDev/";
    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

//    train_dir = "/shared/corpora/ner/wikifier-features/tl/train-camera3/";
//    test_dir = "/shared/corpora/ner/wikifier-features/tl/test-camera3/";


    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->read_good_features("good_features_0.15_fw3");
//    extractor->form_context_size = 3;
//    extractor->brown_context_size = 3;
//    extractor->gazetteer_context_size = 1;
//    extractor->use_gazetteer = false;
//    extractor->use_brown = false;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
//    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 8;

//    select_features(train_prob, extractor, c, "good_features_0.17_fw3");

    vector<Document *> *test_docs;
    vector<Document *> *dev_docs;
    ofstream paramfile("tmp2");

    for(int i = 0; i < 10; i++){
        double gamma = 1;
        for(int j = 0; j < 1; j++) {
            double coef = 8;
            for (int k = 0; k < 1; k++) {

                cout << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);

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


void separate_prob_exp(){

    set_conll_types();

    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

    FeatureExtractor *extractor = new FeatureExtractor();
//    extractor->read_good_features1("tmp_0.5");
    extractor->form_context_size = 3;
//    extractor->brown_context_size = 3;
//    extractor->gazetteer_context_size = 1;
    extractor->use_gazetteer = false;
    extractor->use_brown = false;
//    extractor->use_affixe = false;
//    extractor->use_tagpattern = false;
//    extractor->use_tagcontext = false;
//    extractor->use_form = false;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    struct problem **train_prob = build_svm_problem1(train_dir, extractor);


    double c = 0.5;

    // use single problem to select features for all classifiers
//    struct problem *train_prob = build_svm_problem(train_docs, extractor);
//    select_features1(train_prob, extractor, c, "tmp_0.5");

    extractor->training = false;
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
                struct model **models = train_ner1(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);

                // dev
                dev_docs = readColumnFormatFiles(dev_dir);
                clean_cap_words(dev_docs);
                predict_ner1(dev_docs, extractor, models);
                bilou_to_bio(dev_docs);
                double f1_dev = evaluate_phrases(dev_docs);
                free_docs(dev_docs);

                // test
                test_docs = readColumnFormatFiles(test_dir);
                clean_cap_words(test_docs);
                predict_ner1(test_docs, extractor, models);
                free_models(models);
                bilou_to_bio(test_docs);
                double f1 = evaluate_phrases(test_docs);
                paramfile << f1_dev << " " << f1 << endl;
                coef /= 2;
                free_docs(test_docs);
            }
            gamma /= 2;
        }
        c/=2;
    }
    paramfile.close();
}

void run_sota(){

    set_conll_types();

    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

    FeatureExtractor *extractor = new FeatureExtractor();
    extractor->read_good_features("good_features_0.15_fw3");
    extractor->form_context_size = 3;
    extractor->form_conj = false;

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
//    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 0.5, gamma = 1, coef = 8;

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

void am_exp(){

#ifndef WEIGHT
    cout << "Use weighted SVMs to get better performance!!!" << endl;
    exit(-1);
#endif

//    set_conll_types();

    set_lorelei_types();

    const char *train_dir = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/final-test2-stem-wiki/";
    //const char *train_dir = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/dryrun-outputs/rpi/Train-full-stem/";
    //const char *train_dir = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/uiuc+rpi/";
//    const char *train_dir = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/final-test2.bak-stem/";
//    const char *train_dir = "/shared/corpora/ner/translate/am/Train-stem/";
//    const char *test_dir = "/shared/corpora/ner/lorelei/am/All-nosn-stem/";
    const char *test_dir = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/All-nosn-stem-wiki/";



    FeatureExtractor *extractor = new FeatureExtractor();
    extractor->form_context_size = 3;
    extractor->brown_context_size = 1;
    extractor->gazetteer_context_size = 3;
    extractor->use_tagpattern = false;
//    extractor->use_gazetteer = false;
//    extractor->use_brown = false;
    extractor->gazetteer_list = "/home/ctsai12/CLionProjects/NER/gazetteers-list.am.txt";
//    extractor->use_affixe = false;
//    extractor->use_tagcontext = false;
    extractor->use_cap = false;
//    extractor->use_sen = false;
//    extractor->use_wikifier = true;
    extractor->min_word_freq = 1;
    extractor->prefix_len.push_back(20);
    extractor->brown_cluster_paths.clear();
    extractor->brown_cluster_paths.push_back("/home/mayhew2/software/brown-cluster-master/amdump-c1000-p1.out/paths");
//            "/home/mayhew2/software/brown-cluster-master/amdump-c100-p1.out/paths",
    extractor->brown_cluster_paths.push_back("/home/mayhew2/software/brown-cluster-master/amdump-c500-p1.out/paths");
    extractor->brown_cluster_paths.push_back("/shared/experiments/ctsai12/workspace/brown-cluster/all.plain-c100-p1.out/paths");
    extractor->brown_cluster_paths.push_back("/shared/experiments/ctsai12/workspace/brown-cluster/all.plain-c200-p1.out/paths");
    extractor->brown_cluster_paths.push_back("/shared/experiments/ctsai12/workspace/brown-cluster/all.plain-c500-p1.out/paths");
    extractor->brown_cluster_paths.push_back("/shared/experiments/ctsai12/workspace/brown-cluster/all.plain-c1000-p1.out/paths");


    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
//    const char *mono = "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/conll-sub-stem/";
//    vector<Document *> *train_docs1 = readColumnFormatFiles(mono);

//    vector<Document *> *train_docs2 = readColumnFormatFiles(train_dir);
    //vector<Document *> *en_docs = readColumnFormatFiles("/shared/corpora/ner/translate/am/Train-tac-stem/");
	vector<Document *> *en_docs = readColumnFormatFiles("/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/dryrun-outputs/rpi/Train-full-stem/");
    for(int i = 0; i < en_docs->size(); i++)
        train_docs->push_back(en_docs->at(i));
//    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
//    bio_to_bilou(train_docs1);
//    generate_features(train_docs, extractor);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);


//    struct model *model1 = train_ner(train_prob, L2R_L2LOSS_SVC, 0.015625, 0.1, 1, 1);
//    struct problem *train_prob1 = build_selftrain_problem1(train_docs1, train_docs, extractor, model1);

//    struct model *model2 = train_ner(train_prob1, L2R_L2LOSS_SVC, 0.03125, 0.1, 1, 1);
//    struct problem *train_prob2 = build_selftrain_problem(train_docs2, train_docs1, extractor, model2);

//    select_features(train_prob, extractor, c, "good_features_0.17_fw3");
    extractor->training = false;

    vector<Document *> *test_docs;
    ofstream paramfile("tmp");

    double c = 0.125;

    for(int i = 0; i < 8; i++){
        double gamma = 1;
        for(int j = 0; j < 1; j++) {
            double coef = 8;
            for (int k = 0; k < 1; k++) {

                cout << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);
                cout << "# features " << model->nr_feature << endl;

                // test
                test_docs = readColumnFormatFiles(test_dir);
//                clean_cap_words(test_docs);
                predict_ner(test_docs, extractor, model);
                free_and_destroy_model(&model);
                bilou_to_bio(test_docs);
                double f1 = evaluate_phrases(test_docs);
                paramfile << "1" << " " << f1 << endl;
//                coef /= 2;
                coef -= 1;

//                writeColumnFormatFiles(test_docs, "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/dryrun-outputs/manual/column/");
//                writeTACFormat(test_docs, "/shared/corpora/corporaWeb/lorelei/data/LDC2016E86_LORELEI_Amharic_Representative_Language_Pack_Monolingual_Text_V1.1/data/monolingual_text/zipped/dryrun-outputs/manual/edl", "NAM");
                free_docs(test_docs);
            }
//            gamma -= 0.25;
            gamma /= 2;
        }
        c/=2;
    }
    paramfile.close();
}

int main() {

//    semisup_exp();
//    conll_exp();
//    separate_prob_exp(); // still buggy
//    nominal_exp();
//    pronominal_exp();

//    am_exp();

    // this should use liblinear-poly2 library, not the one inside this project
    run_sota();

    return 0;
}
