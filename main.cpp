#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <gzstream.h>
#include <linear-poly2.h>
//#include <linear.h>
#include "Document.h"
#include "dirent.h"
#include "FeatureExtractor.h"

using namespace std;

unordered_map<string, string> label2id;
unordered_map<int, string> id2label;
const string types[] = {"LOC", "ORG", "PER", "MISC"};
struct feature_node *x_space;
void print_null(const char *s) {}

/**
 * Read column-format files in the directory
 * @param directory
 * @return
 */
vector<Document *> * readColumnFormatFiles(const char *directory) {

    cout << "Reading " << directory << endl;

    vector<Document *> *docs = new vector<Document *>;

    struct dirent *ent;
    DIR *dir = opendir(directory);
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue; // skip '.' and '..'
        string file_path = string(directory) + ent->d_name;

        Document *doc = new Document(ent->d_name);
        doc->sentences = new vector<Sentence *>;
        Sentence *sentence = new Sentence;

        ifstream infile(file_path);
        string line;
        while (getline(infile, line)) {
            if (line.empty()) {
                if (sentence->size() > 0) {
                    doc->sentences->push_back(sentence);
                    sentence = new Sentence;
                }
                continue;
            }
            vector<string> *wiki_feats = new vector<string>;
            stringstream ss(line);
            string buf, surface, label;
            int c = 0;
            while (ss >> buf) {
                if (c == 0)
                    label = buf;
                if (c == 5)
                    surface = buf;
                if (c >= 10)
                    wiki_feats->push_back(buf);
                c++;
            }
            Token *token = new Token(surface, label);
            token->wikifier_features = wiki_feats;
            sentence->tokens->push_back(token);
        }
        docs->push_back(doc);
    }
    closedir (dir);
    return docs;
}

bool mix_case(string &word){
    bool low = false, up = false;
    for(char &c: word) {
        if (islower(c)) low = true;
        if (isupper(c)) up = true;
    }
    return low && up;
}

bool all_low(string &word){
    bool up = false;
    for(char &c: word)
        if (isupper(c)) up = true;
    return not up;
}

bool all_alpha(string &word){
    bool all = true;
    for(char &c: word)
        if (!isalpha(c)) all = false;
    return all;
}

bool equal_ignore_case(string &word1, string &word2){
    if(word1.size() != word2.size()) return false;
    for(int i = 0; i < word1.size(); i++)
        if(tolower(word1[i]) != tolower(word2[i]))
            return false;
    return true;
}

bool all_cap(string &word){
    bool allcap = true;
    for(char c: word)
        if(islower(c))
            return false;
    return allcap;
}

/**
 * Fix words with all capitalized letters if there is any mix-cased version in the document
 * @param docs
 */
void clean_cap_words(vector<Document *> *docs){

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        unordered_map<string, string> cap2norm;
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int i = 0; i < sen->size(); i++){
                Token *token = sen->get_token(i);
                if(!all_alpha(token->surface) || token->surface.size()<=3) continue;

                if(all_cap(token->surface)){
                    bool got = false;
                    for(int sen_id1 = 0; sen_id1 < doc->size(); sen_id1++){
                        Sentence *sen1 = doc->get_sentence(sen_id);
                        for(int j = 0; j < sen1->size(); j++){
                            Token *token1 = sen1->get_token(j);
                            if(token->surface != token1->surface && equal_ignore_case(token->surface, token1->surface)) {
                                if (all_low(token1->surface)) {
                                    token->surface = token1->surface;
                                    got = true;
                                    break;
                                } else if (mix_case(token1->surface)) {
                                    token->surface = token1->surface;
                                }
                            }
                        }
                        if(got) break;
                    }
                }
            }
        }
    }
}

/**
 * Convert BIO labeling scheme to BILOU
 * @param docs
 */
void bio_to_bilou(vector<Document *> *docs){

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int i = 0; i < sen->size(); i++){
                Token *token = sen->get_token(i);
                char next_label = '-';
                if(i < sen->size()-1)
                    next_label = sen->get_token(i+1)->label[0];

                if(token->label[0] == 'B' && (next_label == '-' || next_label != 'I'))
                        token->label[0] = 'U';
                else if(token->label[0] == 'I' && (next_label == '-' || next_label == 'O'))
                        token->label[0] = 'L';
            }
        }
    }
}

/**
 * Convert BILOU labeling scheme to BIO
 * @param docs
 */
void bilou_to_bio(vector<Document *> *docs){
    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int i = 0; i < sen->size(); i++){
                Token *token = sen->get_token(i);
                if(token->prediction[0] == 'U')
                    token->prediction[0] = 'B';
                else if(token->prediction[0] == 'L')
                    token->prediction[0] = 'I';
            }
        }
    }
}

/**
 * Extract features and counstruct SVM problem
 * @param docs
 * @param extractor
 * @return SVM problem
 */
struct problem * build_svm_problem(vector<Document *> *docs, FeatureExtractor *extractor){
    cout << "Building training instances..." << endl;

    struct problem *prob = new problem();

    prob->l = 0;
    int element = 0;

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        extractor->gen_brown_cache(doc);
        extractor->gen_gazetteer_cache(doc);
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int tok_id = 0; tok_id < sen->size(); tok_id++){
                extractor->extract(doc, sen_id, tok_id);
                prob->l++;
                element += doc->get_token(sen_id, tok_id)->features->size() + 1;
            }
        }
    }

    prob->bias = 1;

    prob->y = (double *) malloc(prob->l * sizeof(double));
    prob->x = (struct feature_node **) malloc(prob->l * sizeof(struct feature_node));
    x_space = (struct feature_node *) malloc((element+prob->l) * sizeof(struct feature_node));

    int max_index = 0;
    int ins = 0;
    int fea = 0;

//    ofstream outfile("tmp");

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        for(int sen_id = 0; sen_id < docs->at(doc_id)->size(); sen_id++){
            for(int tok_id = 0; tok_id < docs->at(doc_id)->sentence_size(sen_id); tok_id++){
                Token *token = docs->at(doc_id)->get_token(sen_id, tok_id);

                int label = stoi(label2id[token->label]);

                prob->x[ins] = &x_space[fea];
                prob->y[ins] = label;

//                outfile << label;
                vector<int> features;
                unordered_map<int, double>::iterator it;
                for (it = token->features->begin(); it != token->features->end(); ++it)
                    features.push_back((*it).first);
                sort(features.begin(), features.end());
                for (int f: features) {
                    x_space[fea].index = f;
                    x_space[fea].value = token->features->at(f);
//                    outfile << " "+to_string(f)+":"+to_string(token.features[f]);
                    fea++;
                }
//                outfile << "\n";
                max_index = max(max_index, features.back());

                if(prob->bias >= 0)
                    x_space[fea++].value = prob->bias;

                x_space[fea++].index = -1;

                ins++;
            }
        }
    }
//    outfile.close();

    if(prob->bias >= 0)
    {
        prob->n=max_index+1;
        for(int i=1;i<prob->l;i++)
            (prob->x[i]-2)->index = prob->n;
        x_space[fea-2].index = prob->n;
    }
    else
        prob->n=max_index;

    cout << "#training instances " << prob->l << endl;
    return prob;
}

/**
 * Run liblinear on the input problem
 * @param prob
 * @param solver
 * @param c
 * @param eps
 * @param gamma
 * @param coef
 * @return
 */
model* train_ner(struct problem *prob,int solver, double c, double eps, double gamma, double coef){

    struct parameter param;
    param.C = c;
    param.solver_type = solver;
    param.init_sol = NULL;
    param.weight = NULL;
    param.weight_label = NULL;
    param.eps = eps;
    param.p = 0.1;
    param.nr_weight = 0;
#ifdef POLY2
    param.gamma =gamma;
    param.coef0 = coef;
    prob->gamma = param.gamma;
    prob->coef0 = param.coef0;
#endif

    set_print_string_function(&print_null);
    struct model* model = train(prob, &param);

    destroy_param(&param);

    return model;
}

/**
 * Using the input model to make predictions on the input documents
 * @param docs
 * @param extractor
 * @param model
 */
void predict_file(vector<Document *> *docs, FeatureExtractor *extractor, model* model){

    cout << "Predicting..." << endl;

    struct feature_node *x;
    int max_nr_attr = 64;
    x = (struct feature_node *) malloc(max_nr_attr * sizeof(struct feature_node));
    int nr_feature=get_nr_feature(model);
    int n;
    if(model->bias>=0)
        n=nr_feature+1;
    else
        n=nr_feature;

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        extractor->gen_brown_cache(doc);
        extractor->gen_gazetteer_cache(doc);
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int tok_id = 0; tok_id < sen->size(); tok_id++){
                extractor->extract(doc, sen_id, tok_id);
                Token *token = doc->get_token(sen_id, tok_id);
                vector<int> features;
                unordered_map<int, double>::iterator it;
                for (it = token->features->begin(); it != token->features->end(); ++it)
                    features.push_back((*it).first);
                sort(features.begin(), features.end());

                int k = 0;
                for(int f: features) {
                    if (k >= max_nr_attr - 2)    // need one more for index = -1
                    {
                        max_nr_attr *= 2;
                        x = (struct feature_node *) realloc(x, max_nr_attr * sizeof(struct feature_node));
                    }
                    x[k].index = f;
                    x[k].value = token->features->at(f);

                    if(x[k].index <= nr_feature)
                        k++;
                }
                if(model->bias>=0)
                {
                    x[k].index = n;
                    x[k].value = model->bias;
                    k++;
                }
                x[k].index = -1;

                int predict_label = (int)predict(model, x);

                token->prediction = id2label[predict_label];
            }
        }
    }

    free(x);
}

//void evaluate_tokens(vector<Document> &docs){
//
//    unordered_map<string, int> gold_token;
//    unordered_map<string, int> pred_token;
//    unordered_map<string, int> corr_token;
//    for(string type: types) {
//        gold_token[type] = 0;
//        pred_token[type] = 0;
//        corr_token[type] = 0;
//    }
//
//    for(Document &doc: docs){
//        for(Sentence &sentence: doc.sentences){
//            for(Token &token: sentence.tokens){
//
//                const string &gold_type = token.get_gold_type();
//                const string &pred_type = token.get_pred_type();
//
//                if(gold_type != "O")
//                    gold_token.find(gold_type)->second++;
//
//                if(pred_type != "O")
//                    pred_token.find(pred_type)->second++;
//
//                if(gold_type != "O" && gold_type == pred_type)
//                    corr_token.find(pred_type)->second++;
//            }
//        }
//    }
//
//    printf("=============== Token level ================\n");
//    int total_corr_token = 0, total_pred_token = 0, total_gold_token = 0;
//    for(string type: types){
//        total_corr_token += corr_token.find(type)->second;
//        total_pred_token += pred_token.find(type)->second;
//        total_gold_token += gold_token.find(type)->second;
//        double pre = ((double)corr_token.find(type)->second) / pred_token.find(type)->second;
//        double rec = ((double)corr_token.find(type)->second) / gold_token.find(type)->second;
//        double f1 = 2*pre*rec/(pre+rec);
//        printf("%s precision %.2f recall %.2f f1 %.2f %d %d %d\n", type.c_str(), pre*100, rec*100, f1*100, corr_token[type], pred_token[type], gold_token[type]);
//    }
//
//    double pre = ((double)total_corr_token)/total_pred_token;
//    double rec = ((double)total_corr_token)/total_gold_token;
//    printf("Overall precision %.2f recall %.2f f1 %.2f\n", pre*100, rec*100, 200*pre*rec/(pre+rec));
//
//}

double evaluate_phrases(vector<Document *> *docs){

    unordered_map<string, int> gold_cnt;
    unordered_map<string, int> pred_cnt;
    unordered_map<string, int> corr_cnt;
    for(string type: types) {
        gold_cnt[type] = 0;
        pred_cnt[type] = 0;
        corr_cnt[type] = 0;
    }

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            int gold_start = -1, pred_start = -1;
            string gold_ptype = "", pred_ptype = "";
            for(int tok_id = 0; tok_id < sen->size(); tok_id++){
                Token *token = doc->get_token(sen_id, tok_id);
                bool get_gold = false, get_pred = false;
                string gold_type = token->get_gold_type();
                if (gold_start > -1 &&
//                    (token.label == "O" || (token.label[0] == 'B' && gold_type!=gold_ptype) || gold_type != gold_ptype)) { // end of a phrase
                    (token->label == "O" || (token->label[0] == 'B') || gold_type != gold_ptype)) { // end of a phrase

                    get_gold = true;
                    gold_cnt.find(gold_ptype)->second++;
                }

                string pred_type = token->get_pred_type();
                if (pred_start > -1 &&
//                    (token.prediction == "O" || (token.prediction[0] == 'B' && pred_type!=pred_ptype) || pred_type != pred_ptype)) { // end of a phrase
                    (token->prediction == "O" || (token->prediction[0] == 'B' ) || pred_type != pred_ptype)) { // end of a phrase
//                    (token.prediction == "O" || (token.prediction[0] == 'B'))) { // end of a phrase

                    get_pred = true;
                    pred_cnt.find(pred_ptype)->second++;
//                    string phrase = sentence.tokens.at(pred_start).surface;
//                    for(int k = pred_start+1; k < i; k++)
//                        phrase += " "+sentence.tokens.at(k).surface;
//                    cout << phrase << endl;
                }

                if (get_gold && get_pred && gold_ptype == pred_ptype && gold_start == pred_start) {
                    corr_cnt.find(pred_ptype)->second++;
                }

                if (token->label[0] == 'B')
                    gold_start = tok_id;
                else if (token->label == "O" || gold_type != gold_ptype)
                    gold_start = -1;
//                if (token.label == "O")
//                    gold_start = -1;
//                else if (token.label[0] == 'B' || gold_type!=gold_ptype)
//                    gold_start = i;

                if (token->prediction[0] == 'B')
                    pred_start = tok_id;
                else if (token->prediction == "O" || pred_type != pred_ptype)
                    pred_start = -1;

//                if(token.prediction == "O")
//                    pred_start = -1;
//                else if (token.prediction[0] == 'B' || pred_type != pred_ptype)
//                    pred_start = i;

                gold_ptype = gold_type;
                pred_ptype = pred_type;
            }

            // handle the phrases at the end of sentences
            if (gold_start > -1)
                gold_cnt.find(gold_ptype)->second++;

            if (pred_start > -1)
                pred_cnt.find(pred_ptype)->second++;

            if (gold_start>-1 && pred_start>-1 && gold_ptype == pred_ptype && gold_start == pred_start)
                corr_cnt.find(pred_ptype)->second++;
        }
    }

    printf("=============== Phrase level ================\n");
    int total_corr = 0, total_pred = 0, total_gold = 0;
    for(string type: types){
        total_corr += corr_cnt.find(type)->second;
        total_pred += pred_cnt.find(type)->second;
        total_gold += gold_cnt.find(type)->second;
        double pre = ((double)corr_cnt.find(type)->second) / pred_cnt.find(type)->second;
        double rec = ((double)corr_cnt.find(type)->second) / gold_cnt.find(type)->second;
        double f1 = 2*pre*rec/(pre+rec);
        printf("%s precision %.2f recall %.2f f1 %.2f %d %d %d\n", type.c_str(), pre*100, rec*100, f1*100, corr_cnt[type], pred_cnt[type], gold_cnt[type]);
    }

    double pre = ((double)total_corr)/total_pred;
    double rec = ((double)total_corr)/total_gold;
    double f1 = 2*pre*rec/(pre+rec);
    printf("Overall precision %.2f recall %.2f f1 %.2f\n", pre*100, rec*100, f1*100);

    return f1*100;
}

int main() {


    int cnt = 0;
    for(string type: types){
        label2id["B-"+type] = to_string(cnt);
        id2label[cnt++] = "B-"+type;
        label2id["I-"+type] = to_string(cnt);
        id2label[cnt++] = "I-"+type;
        label2id["L-"+type] = to_string(cnt);
        id2label[cnt++] = "L-"+type;
        label2id["U-"+type] = to_string(cnt);
        id2label[cnt++] = "U-"+type;
    }
    label2id["O"] = to_string(cnt);
    id2label[cnt] = "O";

//    const char *train_dir = "/shared/corpora/ner/wikifier-features/en/train-camera3/";
//    const char *test_dir = "/shared/corpora/ner/wikifier-features/en/test-camera3/";

//    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/TrainPlusDev/";
    const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train/";
    const char *dev_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Dev/";
    const char *test_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Test/";

    FeatureExtractor *extractor = new FeatureExtractor();

    vector<Document *> *train_docs = readColumnFormatFiles(train_dir);
    clean_cap_words(train_docs);
    bio_to_bilou(train_docs);
    struct problem *train_prob = build_svm_problem(train_docs, extractor);

    extractor->training = false;

    double c = 0.0025;
//    struct model *model = train_ner(train_prob, L1R_L2LOSS_SVC, c, 0.01,0,0);
//
//    ofstream outfile("good_features_0.25");
//
//    for(int i = 0; i < model->nr_feature; i++){
//        for(int j = 0; j < model->nr_class; j++) {
//            double w = model->w[i*model->nr_class+j];
//            if(w != 0){
//                int fid = i+1;
//                string fname = extractor->id2feature->at(fid);
//                outfile << to_string(fid) +"\t"+fname+"\t"+to_string(w)+"\t"+to_string(j)<< endl;
//            }
//        }
//    }
//
//    outfile.close();
//    exit(-1);

    vector<Document *> *test_docs;
    vector<Document *> *dev_docs;
    ofstream paramfile("param_search_0.25_4");

    double max_f1 = 0;
    for(int i = 0; i < 1; i++){
        double gamma = 4;
        for(int j = 0; j < 5; j++) {
            double coef = 4;
            for (int k = 0; k < 5; k++) {

                cout << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                paramfile << "c " << c << " gamma " << gamma << " coef " << coef << endl;
                struct model *model = train_ner(train_prob, L2R_L2LOSS_SVC, c, 0.1, gamma, coef);
                cout << "#features " << model->nr_feature << endl;
                dev_docs = readColumnFormatFiles(dev_dir);
                test_docs = readColumnFormatFiles(test_dir);
                clean_cap_words(dev_docs);
                clean_cap_words(test_docs);
                predict_file(dev_docs, extractor, model);
                predict_file(test_docs, extractor, model);
                free_and_destroy_model(&model);
                bilou_to_bio(dev_docs);
                bilou_to_bio(test_docs);
                double f1_dev = evaluate_phrases(dev_docs);
                double f1 = evaluate_phrases(test_docs);
                paramfile << f1_dev << " " << f1 << endl;
                max_f1 = max(max_f1, f1);
                coef /= 2;
                free(dev_docs);
                free(test_docs);
            }
            gamma /= 2;
        }
        c/=2;
    }
    cout << "max f1 " << max_f1 << endl;
//    paramfile << "max f1 " << max_f1 << endl;
    paramfile.close();

    return 0;
}