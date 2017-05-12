//
// Created by Chen-Tse Tsai on 5/11/17.
//

#include <complex>
#include <iostream>
#include <fstream>
#include "FeatureExtractor.h"

#define BOOL_STR(b) ((b)?"true":"false")

void FeatureExtractor::add_feature(Token &token, string feature) {

    int fid = get_feature_id(feature);
    token.features.insert(fid);

}

int FeatureExtractor::get_feature_id(string name) {

    if(feature2id.find(name) == feature2id.end()){
        feature2id.insert({name, fcnt});
        fcnt++;
    }
    unordered_map<string, int>::iterator it = feature2id.find(name);

    return it->second;
}

void FeatureExtractor::sentence_start(Sentence &sentence, int idx) {

    if(idx == 0){
        add_feature(sentence.tokens.at(0), "SentenceStart");
    }
}

void FeatureExtractor::capitalization(Sentence &sentence, int idx) {

    Token &target = sentence.tokens.at(idx);
    for(int j = -2; j <= 2; j++){
        int i = idx + j;
        if(i >= 0 && i < sentence.tokens.size()){
            if(sentence.tokens.at(i).capitalized){
                add_feature(target, to_string(j)+":isCapitalized");
            }
        }
    }
}

void FeatureExtractor::forms(Sentence &sentence, int idx) {
    Token &target = sentence.tokens.at(idx);
    for(int j = -2; j <= 2; j++){
        int i = idx + j;
        if(i >= 0 && i < sentence.tokens.size()){
            add_feature(target, to_string(j)+":"+sentence.tokens.at(i).surface);

            string norm = normalize_digits(sentence.tokens.at(i).surface);
            add_feature(target, to_string(j)+":"+norm);
        }
    }
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

void FeatureExtractor::word_type(Sentence &sentence, int idx) {
    Token &target = sentence.tokens.at(idx);
    for(int j = -2; j <= 2; j++){
        int i = idx + j;
        if(i >= 0 && i < sentence.tokens.size()){

            bool all_cap = true, all_digit = true, all_nonletter = true;

            for(int k = 0; k < sentence.tokens.at(i).surface.size(); k++){
                char c = sentence.tokens.at(i).surface[k];
                all_cap &= isupper(c);
                all_digit &= (isdigit(c) || c=='.' || c==',');
                all_nonletter &= !isalpha(c);
            }

            add_feature(target, to_string(j)+":allcap:"+BOOL_STR(all_cap));
            add_feature(target, to_string(j)+":alldigit:"+BOOL_STR(all_digit));
            add_feature(target, to_string(j)+":allnonletter:"+BOOL_STR(all_nonletter));
        }
    }

}

void FeatureExtractor::affixes(Sentence &sentence, int idx) {
    Token &target = sentence.tokens.at(idx);
    string surface = target.surface;
    int n = surface.size();
    for(int j = 3; j < 5; j++){
        if(n >= j) {
            add_feature(target, "prefix:" + surface.substr(0, j));
        }
    }

    for(int j = 1; j < 5; j++){
        if(n >= j){
            add_feature(target, "suffix:" + surface.substr(n - j, j));
        }
    }
}

void FeatureExtractor::previous_tag1(Sentence &sentence, int idx) {

    if(idx > 0) {
        Token &target = sentence.tokens.at(idx);
        add_feature(target, "pretag:" + sentence.tokens.at(idx - 1).label);
    }
}

void FeatureExtractor::previous_tag2(Sentence &sentence, int idx) {

    if(idx > 1) {
        Token &target = sentence.tokens.at(idx);
        add_feature(target, "prepretag:" + sentence.tokens.at(idx - 2).label);
    }
}

void FeatureExtractor::extract(Sentence &sentence, int idx) {

    sentence_start(sentence, idx);

    capitalization(sentence, idx);

    forms(sentence, idx);

    affixes(sentence, idx);

    word_type(sentence, idx);

    previous_tag1(sentence, idx);

    previous_tag2(sentence, idx);

}

void FeatureExtractor::save_feature_map(string filepath) {


    ofstream ffile (filepath);

    for(auto it = feature2id.begin(); it != feature2id.end(); ++it){
        ffile << it->first +"\t"+to_string(it->second) +"\n";
    }

    ffile.close();

}

void FeatureExtractor::read_feature_map(string filepath) {

    feature2id.clear();

    ifstream infile(filepath);
    string line, key, value;
    int max_idx = 0;
    int cnt = 0;
    while (getline(infile, line)) {
        stringstream ss(line);
        ss >> key >> value;
        int fid = stoi(value);
        max_idx = max(max_idx, fid);
        feature2id.insert({key, fid});
        cnt++;
    }

    fcnt = max_idx+1;

    infile.close();

    cout << "read "+to_string(cnt)+" features" << endl;
}


