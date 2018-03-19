//
// Created by ctsai12 on 5/30/17.
//

#include "Utils.h"
void print_null(const char *s) {}

/**
 * Read column-format files in the director
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
            istringstream ss(line);
            string buf, surface, label;
            int start = -1, end = -1;
            int c = 0;
//            while (ss >> buf) {
            while (getline(ss, buf, '\t')) {
                if (c == 0)
                    label = buf;
                if (c == 2 && isdigit(buf[0])) // the start offset of the token
                    start = stoi(buf);
                if (c == 3 && isdigit(buf[0]))
                    end = stoi(buf);
                if (c == 5) {
                    stringstream sss(buf);
                    sss >> buf;
                    surface = buf;
                }
                if (c >= 10)
                    wiki_feats->push_back(buf);
                c++;
            }
            if(surface == "-DOCSTART-") continue;

            if(surface == "-LRB-")
                surface = "(";
            else if(surface == "-RRB-")
                surface = ")";
            Token *token = new Token(surface, label);
            token->wikifier_features = wiki_feats;
            sentence->tokens->push_back(token);
            if(start != -1 && end != -1){
                token->start_offset = start;
                token->end_offset = end;
            }
        }
        if (sentence->size() > 0) {
            doc->sentences->push_back(sentence);
            sentence = new Sentence;
        }
        docs->push_back(doc);
    }
    closedir (dir);
    cout << "  Read " << docs->size() << " docs" << endl;
    return docs;
}

void writeColumnFormatFiles(vector<Document *> *docs, string dir) {

    string command = "mkdir -p "+dir;

    system(command.c_str());

    cout << "Writing documents to " << dir << endl;

    for(int i = 0; i < docs->size(); i++){
        Document *doc = docs->at(i);
        cout << doc->id << endl;
        ofstream outfile(dir+"/"+doc->id);
        for(int j = 0; j < doc->sentences->size(); j++){
            Sentence *sen = doc->get_sentence(j);
            for(int k = 0; k < sen->size(); k++){
                Token *token = doc->get_token(j, k);
                if(token->start_offset != -1 && token->end_offset != -1)
                    outfile << token->prediction +"\tx\t"+to_string(token->start_offset)+"\t"+to_string(token->end_offset)+"\tx\t"+token->surface+"\tx\tx\tx\tx";
                else
                    outfile << token->prediction +"\tx\tx\tx\tx\t"+token->surface+"\tx\tx\tx\tx";
//                for(auto it = token->features->begin(); it != token->features->end(); it++){
//                    outfile << "\t" << (*it).first << ":" << (*it).second;
//                }
                outfile << endl;
            }
            outfile << endl;
        }

        outfile.close();
    }
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

struct problem * build_weighted_svm_problem(vector<Document *> *docs, FeatureExtractor *extractor, double ow){
    cout << "Building training instances..." << endl;

    extractor->training = true;

    struct problem *prob = new problem();
    struct feature_node *x_space;

    prob->l = 0;
    int element = 0;

    // count number of instances and features
    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        extractor->gen_brown_cache(doc);
        extractor->gen_gazetteer_cache(doc);

        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int tok_id = 0; tok_id < sen->size(); tok_id++){
                extractor->extract(doc, sen_id, tok_id);
                Token *token = doc->get_token(sen_id, tok_id);
                if(token->features->size() == 0) continue;

                element += token->features->size()+1;
                prob->l++;
            }
        }
    }

    prob->bias = 1;
    prob->y = (double *) malloc(prob->l * sizeof(double));
    prob->x = (struct feature_node **) malloc(prob->l * sizeof(struct feature_node));
    x_space = (struct feature_node *) malloc((element+prob->l) * sizeof(struct feature_node));

#ifdef WEIGHT
    prob->W = (double *) malloc(prob->l * sizeof(double));
#endif

    int max_index = 0;
    int ins = 0;
    int fea = 0;

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        for(int sen_id = 0; sen_id < docs->at(doc_id)->size(); sen_id++){
            for(int tok_id = 0; tok_id < docs->at(doc_id)->sentence_size(sen_id); tok_id++){
                Token *token = docs->at(doc_id)->get_token(sen_id, tok_id);
                if(token->features->size() == 0) continue;

                int label = stoi(label2id->at(token->label));

                prob->x[ins] = &x_space[fea];
                prob->y[ins] = label;
#ifdef WEIGHT

                if (token->label == "O")
                    prob->W[ins] = ow;
                else
                    prob->W[ins] = 1;
#endif

                vector<int> features;
                unordered_map<int, double>::iterator it;
                for (it = token->features->begin(); it != token->features->end(); ++it) {
                    int fid = (*it).first;
                    features.push_back(fid);
                }

                sort(features.begin(), features.end());
                for (int f: features) {
                    x_space[fea].index = f;
                    x_space[fea].value = token->features->at(f);
                    fea++;
                }

                if(features.size() == 0)
                    cout << "0 features " << endl;
                else
                    max_index = max(max_index, features.back());

                if(prob->bias >= 0)
                    x_space[fea++].value = prob->bias;

                x_space[fea++].index = -1;

                ins++;

            }
        }
    }

    if(prob->bias >= 0)
    {
        prob->n=max_index+1;
        for(int i=1;i<prob->l;i++)
            (prob->x[i]-2)->index = prob->n;
        x_space[fea-2].index = prob->n;
    }
    else
        prob->n=max_index;

    cout << "Number of instances " << prob->l << endl;
    cout << "Number of features " << prob->n << endl;
    return prob;
}

/**
 * Counstruct SVM problem
 * Note: if weighted SVM is used, the weights are set manually here
 * @param docs
 * @param extractor
 * @return SVM problem
 */
struct problem * build_svm_problem(vector<Document *> *docs, FeatureExtractor *extractor){
    cout << "Building training instances..." << endl;

    struct problem *prob = new problem();
    struct feature_node *x_space;

    prob->l = 0;
    int element = 0;

    // count number of instances and features
    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        Document *doc = docs->at(doc_id);
        extractor->gen_brown_cache(doc);
        extractor->gen_gazetteer_cache(doc);

        for(int sen_id = 0; sen_id < doc->size(); sen_id++){
            Sentence *sen = doc->get_sentence(sen_id);
            for(int tok_id = 0; tok_id < sen->size(); tok_id++){
                extractor->extract(doc, sen_id, tok_id);
                Token *token = doc->get_token(sen_id, tok_id);
                if(token->features->size() == 0) continue;

                element += token->features->size()+1;
                prob->l++;
            }
        }
    }

    prob->bias = 1;
    prob->y = (double *) malloc(prob->l * sizeof(double));
    prob->x = (struct feature_node **) malloc(prob->l * sizeof(struct feature_node));
    x_space = (struct feature_node *) malloc((element+prob->l) * sizeof(struct feature_node));

#ifdef WEIGHT
    prob->W = (double *) malloc(prob->l * sizeof(double));
#endif

    int max_index = 0;
    int ins = 0;
    int fea = 0;

    for(int doc_id = 0; doc_id < docs->size(); doc_id++){
        for(int sen_id = 0; sen_id < docs->at(doc_id)->size(); sen_id++){
            for(int tok_id = 0; tok_id < docs->at(doc_id)->sentence_size(sen_id); tok_id++){
                Token *token = docs->at(doc_id)->get_token(sen_id, tok_id);
                if(token->features->size() == 0) continue;

                int label = stoi(label2id->at(token->label));

                prob->x[ins] = &x_space[fea];
                prob->y[ins] = label;
#ifdef WEIGHT
                if(doc_id > 711) {
                    if (token->label == "O")
                        prob->W[ins] = 0.9;
                    else
                        prob->W[ins] = 1.5;
                }
                else {
                    if (token->label == "O")
                        prob->W[ins] = 0.1;
                    else
                        prob->W[ins] = 1;
                }
#endif

                vector<int> features;
                unordered_map<int, double>::iterator it;
                for (it = token->features->begin(); it != token->features->end(); ++it) {
                    int fid = (*it).first;
                    features.push_back(fid);
                }

                sort(features.begin(), features.end());
                for (int f: features) {
                    x_space[fea].index = f;
                    x_space[fea].value = token->features->at(f);
                    fea++;
                }

                if(features.size() == 0)
                    cout << "0 features " << endl;
                else
                    max_index = max(max_index, features.back());

                if(prob->bias >= 0)
                    x_space[fea++].value = prob->bias;

                x_space[fea++].index = -1;

                ins++;

            }
        }
    }

    if(prob->bias >= 0)
    {
        prob->n=max_index+1;
        for(int i=1;i<prob->l;i++)
            (prob->x[i]-2)->index = prob->n;
        x_space[fea-2].index = prob->n;
    }
    else
        prob->n=max_index;

    cout << "Number of instances " << prob->l << endl;
    cout << "Number of features " << prob->n << endl;
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
model * train_ner(struct problem *prob,int solver, double c, double eps, double gamma, double coef){

    cout << "Training..." << endl;

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
void predict_ner(vector<Document *> *docs, FeatureExtractor *extractor, model *model){

    extractor->training = false;

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

                token->prediction = id2label->at(predict_label);
            }
        }
    }

    free(x);
}

void print_label_stats(vector<Document *> *docs){
    unordered_map<string, int> labelcnt;

    for(int i = 0; i < docs->size(); i++){
        Document *doc = docs->at(i);
        for(int j = 0; j < doc->size(); j++){
            for(int k = 0; k < doc->sentence_size(j); k++){
                Token *t = doc->get_token(j, k);
                if(t->label[0] == 'B')
                    labelcnt[t->get_gold_type()]++;
            }
        }
    }

    cout << "label counts:" << endl;
    for(int i = 0; i < types->size(); i++){
        cout << "  " << types->at(i) << " " << labelcnt.at(types->at(i)) << endl;
    }
}

void clear_features(vector<Document *> *docs){
    for (int i = 0; i < docs->size(); i++)
        docs->at(i)->clear_features();
}

double evaluate_phrases(vector<Document *> *docs){

    unordered_map<string, int> gold_cnt;
    unordered_map<string, int> pred_cnt;
    unordered_map<string, int> corr_cnt;
    vector<string>::iterator it;
    for(it = types->begin(); it != types->end(); ++it) {
        string type = (*it);
        gold_cnt[type] = 0;
        pred_cnt[type] = 0;
        corr_cnt[type] = 0;
    }

    int gold_seg = 0, pred_seg = 0, corr_seg = 0;

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
                    (token->label == "O" || token->label[0] == 'B' || gold_type != gold_ptype)) { // end of a phrase

                    get_gold = true;
                    gold_cnt.find(gold_ptype)->second++;
                    gold_seg++;
                }

                string pred_type = token->get_pred_type();
                if (pred_start > -1 &&
                    //                    (token.prediction == "O" || (token.prediction[0] == 'B' && pred_type!=pred_ptype) || pred_type != pred_ptype)) { // end of a phrase
                    (token->prediction == "O" || (token->prediction[0] == 'B' ) || pred_type != pred_ptype)) { // end of a phrase
//                    (token->prediction == "O" || (token->prediction[0] == 'B'))) { // end of a phrase

                    get_pred = true;
                    pred_cnt.find(pred_ptype)->second++;
                    pred_seg++;
//                    string phrase = sentence.tokens.at(pred_start).surface;
//                    for(int k = pred_start+1; k < i; k++)
//                        phrase += " "+sentence.tokens.at(k).surface;
//                    cout << phrase << endl;
                }

                if (get_gold && get_pred && gold_ptype == pred_ptype && gold_start == pred_start) {
                    corr_cnt.find(pred_ptype)->second++;
                }

                if (get_gold && get_pred && gold_start == pred_start) {
                    corr_seg++;
                }

                if (token->label[0] == 'B')
                    gold_start = tok_id;
                else if (token->label == "O" || gold_type != gold_ptype)
                    gold_start = -1;

                if (token->prediction[0] == 'B')
                    pred_start = tok_id;
                else if (token->prediction == "O" || pred_type != pred_ptype)
                    pred_start = -1;

//                if (token->prediction[0] == 'B')
//                    pred_start = tok_id;
//                else if (token->prediction == "O")
//                    pred_start = -1;
//                else if(token->prediction[0] == 'I'){
//                    if(pred_type != pred_ptype){
////                        pred_start = tok_id;
//                        pred_start = -1;
//
//                    }
//                }

                gold_ptype = gold_type;
                pred_ptype = pred_type;
            }

            // handle the phrases at the end of sentences
            if (gold_start > -1) {
                gold_cnt.find(gold_ptype)->second++;
                gold_seg++;
            }

            if (pred_start > -1) {
                pred_cnt.find(pred_ptype)->second++;
                pred_seg++;
            }

            if (gold_start>-1 && pred_start>-1 && gold_ptype == pred_ptype && gold_start == pred_start)
                corr_cnt.find(pred_ptype)->second++;

            if (gold_start>-1 && pred_start>-1 && gold_start == pred_start)
                corr_seg++;

        }
    }

    printf("=============== Phrase level ================\n");
    int total_corr = 0, total_pred = 0, total_gold = 0;
    for(it = types->begin(); it != types->end(); ++it) {
        string type = *it;
        total_corr += corr_cnt.find(type)->second;
        total_pred += pred_cnt.find(type)->second;
        total_gold += gold_cnt.find(type)->second;
        double pre = ((double)corr_cnt.find(type)->second) / pred_cnt.find(type)->second;
        double rec = ((double)corr_cnt.find(type)->second) / gold_cnt.find(type)->second;
        double f1 = 2*pre*rec/(pre+rec);
        printf("%s precision %.2f recall %.2f f1 %.2f %d %d %d\n", type.c_str(), pre*100, rec*100, f1*100, corr_cnt[type], pred_cnt[type], gold_cnt[type]);
    }

    double pre = ((double)corr_seg)/pred_seg;
    double rec = ((double)corr_seg)/gold_seg;
    double f1 = 2*pre*rec/(pre+rec);
//    printf("Segmentation precision %.2f recall %.2f f1 %.2f\n", pre*100, rec*100, f1*100);

    pre = ((double)total_corr)/total_pred;
    rec = ((double)total_corr)/total_gold;
    f1 = 2*pre*rec/(pre+rec);
    printf("Overall precision %.2f recall %.2f f1 %.2f\n", pre*100, rec*100, f1*100);


    return f1*100;
}

// save good features in a single file
void select_features(struct problem *train_prob, FeatureExtractor *extractor, double c, string out_file){

    extractor->filter_features = false;
#ifdef POLY2
    cout << "poly2 is set in feature selection!" << endl;
    exit(-1);
#endif
    struct model *model = train_ner(train_prob, L1R_L2LOSS_SVC, c, 0.01,0,0);

    ofstream outfile(out_file);

    for(int i = 0; i < model->nr_feature; i++){
        for(int j = 0; j < model->nr_class; j++) {
            double w = model->w[i*model->nr_class+j];
            if(w != 0){
                int fid = i+1;
                string fname = extractor->id2feature->at(fid);
                outfile << to_string(fid) +"\t"+fname+"\t"+to_string(w)+"\t"+to_string(j)<< endl;
            }
        }
    }

    outfile.close();
    exit(-1);
}

// save good features for each sub-model (in multi-class)
void select_features1(struct problem *train_prob, FeatureExtractor *extractor, double c, string out_file){

    extractor->filter_features = false;
#ifdef POLY2
    cout << "poly2 is set in feature selection!" << endl;
    exit(-1);
#endif
    struct model *model = train_ner(train_prob, L1R_L2LOSS_SVC, c, 0.01,0,0);

    for(int i = 0; i < model->nr_class; i++){
        ofstream outfile(out_file+"_"+to_string(model->label[i]));
        for(int j = 0; j < model->nr_feature; j++) {
            double w = model->w[j*model->nr_class+i];
            if(w != 0){
                int fid = j+1;
                string fname = extractor->id2feature->at(fid);
                outfile << to_string(fid) +"\t"+fname+"\t"+to_string(w)+"\t"+to_string(j)<< endl;
            }
        }
        outfile.close();
    }

    exit(-1);
}

void free_docs(vector<Document *> *docs){
    for(int i = 0; i < docs->size(); i++)
        delete docs->at(i);
}

void free_models(struct model **models){
    for(int i = 0; i < label2id->size(); i++){
        free_model_content(models[i]);
        free(models[i]);
    }
    free(models);
}
