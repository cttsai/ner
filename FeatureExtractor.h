//
// Created by Chen-Tse Tsai on 5/11/17.
//

#ifndef NER_FEATUREEXTRACTOR_H
#define NER_FEATUREEXTRACTOR_H


#include "Document.h"
#include <unordered_map>

class FeatureExtractor {

public:
    void extract(Sentence &sentence, int idx);

    void save_feature_map(string filepath);

    void read_feature_map(string filepath);

private:
    void sentence_start(Sentence &sentence, int idx);
    void capitalization(Sentence &sentence, int idx);
    void forms(Sentence &sentence, int idx);
    void word_type(Sentence &sentence, int idx);
    void affixes(Sentence &sentence, int idx);
    void previous_tag1(Sentence &sentence, int idx);
    void previous_tag2(Sentence &sentence, int idx);
    void previous_tag_pattern(Sentence &sentence);
    void previous_tag_context(Sentence &sentence);

    string normalize_digits(string token);
    unordered_map<string, int> feature2id;


    void add_feature(Token &token, string feature);
    int get_feature_id(string name);

    int fcnt = 1;

};


#endif //NER_FEATUREEXTRACTOR_H
