//
// Created by ctsai12 on 5/9/17.
//

#ifndef NER_DOCUMENT_H
#define NER_DOCUMENT_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

using namespace std;

class Document;

class Token {

public:

    Token(string s, string l){
        surface = s;
        label = l;
        capitalized = isupper(surface[0]);
        features = new unordered_map<int, double>;
        gazetteer_features = new vector<string>;
        brown_cluster_features = new vector<string>;
    }

    ~Token(){
        delete brown_cluster_features;
        delete gazetteer_features;
        delete wikifier_features;
        delete features;
    }

    string get_gold_type(){
        if(label.empty() || label == "O")
            return label;
        return label.substr(2, label.size()-2);
    }

    string get_pred_type(){
        if(prediction.empty() || prediction == "O")
            return prediction;
        return prediction.substr(2, prediction.size()-2);
    }

    string surface;
    string label;
    string prediction;
    int start_offset = -1;
    int end_offset = -1;
    bool capitalized;
    unordered_map<int, double> *features;

    unordered_map<string, double> *type2score;

    // feature cache
    vector<string> *brown_cluster_features;
    vector<string> *gazetteer_features;
    vector<string> *wikifier_features;
};

class Sentence {
public:

    Sentence(){
        tokens = new vector<Token *>;
    }

    ~Sentence(){
        for(int i = 0; i < tokens->size(); i++)
            delete tokens->at(i);
    }

    vector<Token *> *tokens;

    int size(){
        return tokens->size();
    }

    Token * get_token(int id){
        return tokens->at(id);
    }

};


class Document {
public:

    Document(string id){
        this->id = id;
        token_label_cnt = new unordered_map<string,  unordered_map<string, int>*>();
    }

    ~Document(){
        delete token_label_cnt;
        for(int i = 0; i < sentences->size(); i++)
            delete sentences->at(i);
    }
    string id;
    vector<Sentence *> *sentences;
    unordered_map<string, unordered_map<string, int>*> *token_label_cnt;

    Sentence * get_sentence(int id){
        return sentences->at(id);
    }

    Token * get_token(int sen_id, int tok_id){
        return sentences->at(sen_id)->tokens->at(tok_id);
    }

    int size(){
        return sentences->size();
    }

    int sentence_size(int sen_id){
        return sentences->at(sen_id)->size();
    }
    
    void clear_features(){
        delete token_label_cnt;
        token_label_cnt = new unordered_map<string,  unordered_map<string, int>*>();
        for(int i = 0; i < size(); i++){
            Sentence *sen = sentences->at(i);
            for(int j = 0; j < sen->size(); j++){
                sen->get_token(j)->features->clear();
            }
        }
    }

};


#endif //NER_DOCUMENT_H
