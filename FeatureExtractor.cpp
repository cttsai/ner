
// Created by Chen-Tse Tsai on 5/11/17.
//

#include <complex>
#include <iostream>
#include <fstream>
#include <gzstream.h>
#include <algorithm>

#include "FeatureExtractor.h"

extern unordered_map<string, string> *label2id;

void lowercase(string &word) ;

void FeatureExtractor::extract(Document *doc, int sen_id, int tok_id) {

    if(use_tagcontext)
        update_doc_stats(doc, sen_id, tok_id);

    if(use_sen)
        sentence_start(doc, sen_id, tok_id);

    if(use_cap)
        capitalization(doc, sen_id, tok_id);

    if(use_form)
        forms(doc, sen_id, tok_id);

    if(use_affixe)
        affixes(doc, sen_id, tok_id);

    if(use_wordtype)
        word_type(doc, sen_id, tok_id);

    if(use_pretag1)
        previous_tag1(doc, sen_id, tok_id);

    if(use_pretag2)
        previous_tag2(doc, sen_id, tok_id);

    if(use_tagpattern)
        previous_tag_pattern(doc, sen_id, tok_id);

    if(use_tagcontext)
        previous_tag_context(doc, sen_id, tok_id);

    if(use_brown)
        brown_cluster(doc, sen_id, tok_id);

    if(use_gazetteer)
        gazetteer(doc, sen_id, tok_id);

    if(use_hyphen)
        hyphen(doc, sen_id, tok_id);

    if(use_wikifier)
        wikifier(doc, sen_id, tok_id);

//    add_dummy_feature(doc, sen_id, tok_id);
}

void FeatureExtractor::add_dummy_feature(Document *doc, int sen_id, int tok_id){
    Token *target = doc->get_token(sen_id, tok_id);
    if(target->features->size() == 0){
        int fid = get_feature_id("SentenceStart");
        target->features->insert({fid, 0});
    }
}

// update the label counts for the previous token
void FeatureExtractor::update_doc_stats(Document *doc, int sen_id, int tok_id) {

    if(tok_id > 0){

        Token *token = doc->get_token(sen_id, tok_id-1);
        string surface = token->surface;
        string label = token->label;
        if(!training)
            label = token->prediction;

        if(doc->token_label_cnt->find(surface) == doc->token_label_cnt->end())
            doc->token_label_cnt->insert({surface, new unordered_map<string, int>});

        unordered_map<string, int> *cnt = doc->token_label_cnt->at(surface);

        if(cnt->find(label) == cnt->end())
            cnt->insert({label, 1});
        else
            cnt->at(label)++;
    }
}

void FeatureExtractor::add_feature(Token *token, string feature, double value) {

    if(filter_features && good_features->find(feature) == good_features->end()) return;

    if(gf_set >= 0 && good_features1 != NULL && good_features1->at(gf_set)->find(feature) == good_features1->at(gf_set)->end()) return;

    int fid = get_feature_id(feature);
    max_idx = max(max_idx, fid);
    token->features->insert({fid, value});

}

int FeatureExtractor::get_feature_id(string name) {

    if(feature2id->find(name) == feature2id->end()){
        feature2id->insert({name, fcnt});
        id2feature->insert({fcnt, name});
        fcnt++;
    }

    return feature2id->at(name);
}

void FeatureExtractor::sentence_start(Document *doc, int sen_id, int tok_id) {

    if(tok_id == 0){
        add_feature(doc->get_token(sen_id, tok_id), "SentenceStart", 1);
    }
}

void FeatureExtractor::capitalization(Document *doc, int sen_id, int tok_id) {

    Token *target = doc->get_token(sen_id, tok_id);
    for(int j = -2; j <= 2; j++){
        int i = tok_id + j;
        if(i >= 0 && i < doc->sentence_size(sen_id)){
            if(doc->get_token(sen_id, i)->capitalized){
                add_feature(target, "Capitalized:"+to_string(j), 1);
            }
        }
    }
}

void FeatureExtractor::forms(Document *doc, int sen_id, int tok_id) {
    Token *target = doc->get_token(sen_id, tok_id);
    for(int j = -form_context_size; j <= form_context_size; j++){
        int i = tok_id + j;
        if(i >= 0 && i < doc->sentence_size(sen_id)){
            string word = doc->get_token(sen_id, i)->surface;
            string norm = normalize_digits(word);
            add_feature(target, to_string(j)+":"+word, 1);
            add_feature(target, to_string(j)+":"+norm, 1);

            // conjunction with previous tag
            if(form_conj && tok_id > 0) {
                string plabel;
                if(training)
                    plabel = doc->get_token(sen_id, tok_id-1)->label;
                else
                    plabel = doc->get_token(sen_id, tok_id-1)->prediction;

                add_feature(target, to_string(j)+":"+word+":"+plabel, 1);
                add_feature(target, to_string(j)+":"+norm+":"+plabel, 1);

            }
        }
    }
//    if(tok_id > 0) {
//        string psurface = doc->get_token(sen_id, tok_id-1)->surface;
//        add_feature(target, "Bigram:"+doc->get_token(sen_id, tok_id)->surface+":"+psurface,1);
//    }
}

string FeatureExtractor::normalize_digits(string token) {

    string ret = "";

    for(int i = 0; i < token.size(); i++){
        if(isdigit(token[i]))
            ret += "*D*";
        else
            ret += token[i];
    }

    return ret;
}

string get_shape(string &word){
    string ret = "";

    for(char c: word){
        if(isalpha(c)){
            if(isupper(c))
                ret += 'X';
            else
                ret += 'x';
        }
        else
            ret += c;
    }
    return ret;
}

void FeatureExtractor::word_type(Document *doc, int sen_id, int tok_id) {
    Token *target = doc->get_token(sen_id, tok_id);
    for(int j = -2; j <= 2; j++){
        int i = tok_id + j;
        if(i >= 0 && i < doc->sentence_size(sen_id)){

            bool all_cap = true, all_digit = true, all_nonletter = true;

            string word = doc->get_token(sen_id, i)->surface;

            for(int k = 0; k < word.size(); k++){
                char c = word[k];
                all_cap &= isupper(c);
                all_digit &= (isdigit(c) || c=='.' || c==',');
                all_nonletter &= !isalpha(c);
            }

            if(all_cap)
                add_feature(target, "WordType:"+to_string(j)+":allcap", 1);
            if(all_digit)
                add_feature(target, "WordType:"+to_string(j)+":alldigit", 1);
            if(all_nonletter)
                add_feature(target, "WordType:"+to_string(j)+":allnonletter", 1);

//            if(j==0) {
//                string shape = get_shape(word);
//                add_feature(target, "WordShape:" + to_string(j) + ":" + shape, 1);
//            }
        }
    }
}

void FeatureExtractor::affixes(Document *doc, int sen_id, int tok_id) {
    Token *target = doc->get_token(sen_id, tok_id);
    string surface = target->surface;
    int n = surface.size();
    for(int j = 3; j < 5; j++){
        if(n >= j) {
            string prefix = surface.substr(0, j);
            add_feature(target, "Prefix:" + prefix, 1);
        }
    }
//    add_feature(target, "Ignorefirst:" + surface.substr(1, n-1), 1);
//    add_feature(target, "Ignorelast:" + surface.substr(0, n-1), 1);

    for(int j = 1; j < 5; j++){
        if(n >= j){
            string suffix = surface.substr(n - j, j);
            add_feature(target, "Suffix:" + suffix, 1);
        }
    }
}

void FeatureExtractor::previous_tag1(Document *doc, int sen_id, int tok_id) {

    if(tok_id > 0) {
        Token *target = doc->get_token(sen_id, tok_id);
        if(training)
            add_feature(target, "PreTag:" + doc->get_token(sen_id, tok_id-1)->label, 1);
        else
            add_feature(target, "PreTag:" + doc->get_token(sen_id, tok_id-1)->prediction, 1);
    }
}

void FeatureExtractor::previous_tag2(Document *doc, int sen_id, int tok_id) {

    if(tok_id > 1) {
        Token *target = doc->get_token(sen_id, tok_id);
        if(training)
            add_feature(target, "PrePreTag:" + doc->get_token(sen_id, tok_id-2)->label, 1);
        else
            add_feature(target, "PrePreTag:" + doc->get_token(sen_id, tok_id-2)->prediction, 1);
    }
}

void FeatureExtractor::previous_tag_context(Document *doc, int sen_id, int tok_id) {

    Token *target = doc->get_token(sen_id, tok_id);
    for(int i = 0; i <= 2 && tok_id+i < doc->sentence_size(sen_id); i++){
        string word = doc->get_token(sen_id, tok_id+i)->surface;

        if(doc->token_label_cnt->find(word) != doc->token_label_cnt->end()) {
            unordered_map<string, int> *label_cnt = doc->token_label_cnt->at(word);
            double total = 0;
            unordered_map<string, int>::iterator it;
            for (it = label_cnt->begin(); it != label_cnt->end(); ++it)
                total += it->second;

            for (it = label_cnt->begin(); it != label_cnt->end(); ++it){
                if(it->second>0)
                    add_feature(target, "PreTagContext:" + to_string(i) + ":" + it->first, it->second / total);
            }
        }
    }
}

void FeatureExtractor::previous_tag_pattern(Document *doc, int sen_id, int tok_id){

    string label = "O";
    vector<string> pattern;

    int i = -1;
    if(tok_id+i >= 0){
        if(training)
            label = doc->get_token(sen_id, tok_id+i)->label;
        else
            label = doc->get_token(sen_id, tok_id+i)->prediction;
    }
    else
        label = "";

    for(int j = 0; j < 2 && !label.empty() && label == "O"; j++){
        pattern.push_back(doc->get_token(sen_id, tok_id+i)->surface);
        i--;
        if(tok_id+i >= 0){
            if(training)
                label = doc->get_token(sen_id, tok_id+i)->label;
            else
                label = doc->get_token(sen_id, tok_id+i)->prediction;
        }
        else
            label = "";
    }

    if(!pattern.empty() && !label.empty() && label!="O"){
        label = label.substr(2, label.size()-2);
        string feature = "PreTagPattern:";
        for(int j = 0; j < pattern.size(); j++)
            feature = pattern.at(j) + "_" + feature;
        feature = label + "_" + feature;
        add_feature(doc->get_token(sen_id, tok_id), feature, 1);
    }
}

//void FeatureExtractor::save_feature_map(string filepath) {
//
//
//    ofstream ffile (filepath);
//
//    for(auto it = feature2id.begin(); it != feature2id.end(); ++it){
//        ffile << it->first +"\t"+to_string(it->second) +"\n";
//    }
//
//    ffile.close();
//
//}
//
//void FeatureExtractor::read_feature_map(string filepath) {
//
//    feature2id.clear();
//
//    ifstream infile(filepath);
//    string line, key, value;
//    int max_idx = 0;
//    int cnt = 0;
//    while (getline(infile, line)) {
//        stringstream ss(line);
//        ss >> key >> value;
//        int fid = stoi(value);
//        max_idx = max(max_idx, fid);
//        feature2id.insert({key, fid});
//        cnt++;
//    }
//
//    fcnt = max_idx+1;
//
//    infile.close();
//
//    cout << "read "+to_string(cnt)+" features" << endl;
//}

void FeatureExtractor::init_brown_clusters() {

    cout << "reading brown clusters..." << endl;

    brown_clusters = new vector<unordered_map<string, string> *>();

    for(int i = 0; i < brown_cluster_paths.size(); i++){

        unordered_map<string, string> *bc = new unordered_map<string, string>();
        ifstream infile(brown_cluster_paths.at(i));
        string line, id, word, cnt;
        while (getline(infile, line)) {
            stringstream ss(line);
            ss >> id >> word >> cnt;
            if(stoi(cnt) >= min_word_freq){
                bc->insert({word, id});
            }
        }
        brown_clusters->push_back(bc);
//        cout << brown_cluster_paths.at(i) << endl;
//        cout << "#words " << brown_clusters.back().size() << endl;
    }

    brown_initialized = true;
}

void FeatureExtractor::gen_brown_cache(Document *doc){
    if(use_brown && !brown_initialized) init_brown_clusters();
    if(!brown_clusters) return;
    for(int sen_id = 0; sen_id < doc->size(); sen_id++){
        for(int tok_id = 0; tok_id < doc->sentence_size(sen_id); tok_id++){
            Token *target = doc->get_token(sen_id, tok_id);
            get_prefix(target);
        }
    }
}

void FeatureExtractor::get_prefix(Token *token){
    string word = token->surface;
    for(int i = 0; i < brown_clusters->size(); i++){
        if(brown_clusters->at(i)->find(word) != brown_clusters->at(i)->end()){
            unordered_map<string, string> *bc = brown_clusters->at(i);
            string path = bc->at(word);
            for(int len: prefix_len){
                if(len <= path.size()){
                    string pre = path.substr(0, len);
                    token->brown_cluster_features->push_back(to_string(i)+":"+to_string(len)+":"+pre);
                }
                else
                    token->brown_cluster_features->push_back(to_string(i)+":"+to_string(len)+":"+path);
            }
        }
    }
}

void FeatureExtractor::brown_cluster(Document *doc, int sen_id, int tok_id) {

    if(!brown_clusters) return;

    Token *target = doc->get_token(sen_id, tok_id);

    for (int j = -brown_context_size; j <= brown_context_size; j++) {
        int i = tok_id + j;
        if (i >= 0 && i < doc->sentence_size(sen_id)) {
            vector<string> *bf = doc->get_token(sen_id, i)->brown_cluster_features;
            for(int p = 0; p < bf->size(); p++) {
                add_feature(target, "Brown:" + to_string(j) + ":" + bf->at(p), 1);

//                if (tok_id > 0) {
//                    string plabel;
//                    if (training)
//                        plabel = doc->get_token(sen_id, tok_id - 1)->label;
//                    else
//                        plabel = doc->get_token(sen_id, tok_id - 1)->prediction;
//
//                    add_feature(target, "BrownPreTag:"+to_string(j) + ":" + bf->at(p) + ":" + plabel, 1);
//                }
            }
        }
    }
}

void FeatureExtractor::init_gazetteers() {

    cout << "Reading gazetteers..." << endl;

    gazetteers = new vector<unordered_set<string> *>;
    gazetteers_nocase = new vector<unordered_set<string> *>;
    gazetteer_names = new vector<string>;

    ifstream infile(gazetteer_list);
    string path;
    while (getline(infile, path)) {

        gazetteer_names->push_back(path);

        unordered_set<string> *gz = new unordered_set<string>;
        unordered_set<string> *gz_nc = new unordered_set<string>;
        igzstream in(path.c_str());
        string line;
        while (getline(in, line))
        {
            gz->insert(line);
            lowercase(line);
            if(line != "in" && line != "on" && line != "us" && line != "or" && line != "am")
                gz_nc->insert(line);
        }

        gazetteers->push_back(gz);
        gazetteers_nocase->push_back(gz_nc);

//        cout << path << " " << gazetteers.back().size() << endl;
    }

    gazetteer_initialized = true;
}

void FeatureExtractor::gazetteer(Document *doc, int sen_id, int tok_id) {

    if(!gazetteers) return;

    Token *target = doc->get_token(sen_id, tok_id);

    for (int j = -gazetteer_context_size; j <= gazetteer_context_size; j++) {
        int i = tok_id + j;
        if (i >= 0 && i < doc->sentence_size(sen_id)) {
            vector<string> *gf = doc->get_token(sen_id, i)->gazetteer_features;
            for(int p = 0; p < gf->size(); p++){
                add_feature(target, "Gazetteers:"+to_string(j)+":"+gf->at(p), 1);
            }
        }
    }
}

void FeatureExtractor::gen_gazetteer_cache(Document *doc) {

    if(use_gazetteer && !gazetteer_initialized) init_gazetteers();

    if(!gazetteers) return;

    for(int sen_id = 0; sen_id < doc->size(); sen_id++){
        for(int tok_id = 0; tok_id < doc->sentence_size(sen_id); tok_id++){
            Token *target = doc->get_token(sen_id, tok_id);
            string expression = target->surface;
            string expression_lower = target->surface;
            lowercase(expression_lower);
            for(int j = 0; j < 7; j++){

                for(int k = 0; k < gazetteer_names->size(); k++){

                    if(gazetteers->at(k)->find(expression) != gazetteers->at(k)->end()){

                        if(j == 0)
                            target->gazetteer_features->push_back("U-"+gazetteer_names->at(k));
                        else{
                            for(int l = 0; l <= j; l++){
                                Token *t = doc->get_token(sen_id, tok_id+l);
                                if(l == 0)
                                    t->gazetteer_features->push_back("B-"+gazetteer_names->at(k));
                                else if(l>0 && l < j)
                                    t->gazetteer_features->push_back("I-"+gazetteer_names->at(k));
                                else
                                    t->gazetteer_features->push_back("L-"+gazetteer_names->at(k));
                            }
                        }
                    }

                    if(gazetteers_nocase->at(k)->find(expression_lower) != gazetteers_nocase->at(k)->end()){

                        if(j == 0)
                            target->gazetteer_features->push_back("U-"+gazetteer_names->at(k)+"(IC)");
                        else{
                            for(int l = 0; l <= j; l++){
                                Token *t = doc->get_token(sen_id, tok_id+l);
                                if(l == 0)
                                    t->gazetteer_features->push_back("B-"+gazetteer_names->at(k)+"(IC)");
                                else if(l>0 && l < j)
                                    t->gazetteer_features->push_back("I-"+gazetteer_names->at(k)+"(IC)");
                                else
                                    t->gazetteer_features->push_back("L-"+gazetteer_names->at(k)+"(IC)");
                            }
                        }
                    }
                } // end gaz

                if(tok_id+j+1 < doc->sentence_size(sen_id)){
                    expression += " "+doc->get_token(sen_id, tok_id+j+1)->surface;
                    string tmp = doc->get_token(sen_id, tok_id+j+1)->surface;
                    lowercase(tmp);
                    expression_lower += " "+tmp;
                }
                else
                    break;

            } // end j
        }
    }
}

void lowercase(string &word) {
    for (int i = 0; i < word.size(); i++) {
        word[i] = tolower(word[i]);
    }
} 

void FeatureExtractor::wikifier(Document *doc, int sen_id, int tok_id){
    Token *target = doc->get_token(sen_id, tok_id);

    for(int i = 0; i < target->wikifier_features->size(); i++){
//        if(f.find("location") != f.npos || f.find("organization") != f.npos || f.find("person") != f.npos)
            add_feature(target, "Wikifier:" + target->wikifier_features->at(i), 1);
    }
}

void FeatureExtractor::read_good_features(string file) {


    filter_features = true;

    good_features = new unordered_set<string>;
    ifstream infile(file);
    string line, buf;
    while (getline(infile, line)) {
        stringstream ss(line);
        int i = 0;
        while (ss >> buf) {
            if(i==1)
                good_features->insert(buf);
            i++;
        }
    }
    cout << "#good features " << good_features->size() << endl;
}

void FeatureExtractor::read_good_features1(string file) {

    filter_features = false;
    good_features1 = new vector<unordered_set<string> *>;

    for(int j = 0; j < label2id->size(); j++) {
        unordered_set<string> *fs = new unordered_set<string>;
        ifstream infile(file+"_"+to_string(j));
//        ifstream infile(file);
        string line, buf;
        while (getline(infile, line)) {
            stringstream ss(line);
            int i = 0;
            while (ss >> buf) {
                if (i == 1)
                    fs->insert(buf);
                i++;
            }
        }
        good_features1->push_back(fs);
        cout << "#good features " << good_features1->back()->size() << endl;
    }
}

void FeatureExtractor::hyphen(Document *doc, int sen_id, int tok_id) {
    Token *target = doc->get_token(sen_id, tok_id);

    for (int j = -hyphen_context_size; j <= hyphen_context_size; j++) {
        int i = tok_id + j;
        if (i >= 0 && i < doc->sentence_size(sen_id)) {
            string &word = doc->get_token(sen_id, i)->surface;
            if(word.find('-') != word.npos) {
                add_feature(target, "ContainsHyphen:" + to_string(j), 1);

                stringstream ss(word);
                string tok;
                while(getline(ss, tok, '-'))
                    add_feature(target, "SubToken:"+to_string(j)+":"+tok, 1);
            }
        }
    }

}

void FeatureExtractor::context_ner(Document *doc, int sen_id, int tok_id) {
    Token *target = doc->get_token(sen_id, tok_id);

    for (int j = -2; j <= 2; j++) {
        int i = tok_id + j;
        if (i >= 0 && i < doc->sentence_size(sen_id)) {
            string ner_tag = doc->get_token(sen_id, i)->wikifier_features->at(0);
            add_feature(target, "ContextNER:"+to_string(j)+":"+ner_tag, 1);

        }
    }
}

void FeatureExtractor::prev_context_ner(Document *doc, int sen_id, int tok_id) {

    string prev_tag = "";
    int dist = 0;

    for(int i = tok_id - 1; i >=0; i--){
        dist++;
        string tag = doc->get_token(sen_id, i)->wikifier_features->at(0);
        if(tag != "O") {
            prev_tag = tag.substr(2, tag.size() - 2);
            break;
        }
    }

    if(prev_tag != ""){
        for(int i = sen_id - 1; i >= 0 && prev_tag == ""; i--){
            for(int j = doc->sentence_size(i); j >= 0; j--){
                dist++;
                string tag = doc->get_token(i, j)->wikifier_features->at(0);
                if(tag != "O") {
                    prev_tag = tag.substr(2, tag.size() - 2);
                    break;
                }
            }
        }
    }

    add_feature(doc->get_token(sen_id, tok_id), "PreviousNERTag:"+to_string(dist)+":"+prev_tag, 1);
}
