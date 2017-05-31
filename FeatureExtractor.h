//
// Created by Chen-Tse Tsai on 5/11/17.
//

#ifndef NER_FEATUREEXTRACTOR_H
#define NER_FEATUREEXTRACTOR_H


#include "Document.h"
#include <unordered_map>

class FeatureExtractor {

public:

    FeatureExtractor(){
        feature2id = new unordered_map<string ,int>();
        id2feature = new unordered_map<int, string>();
        init_gazetteers();
        init_brown_clusters();
        if(filter_features)
            read_good_features("good_features_0.25");
    }
    bool filter_features = true;

    void extract(Document *doc, int sen_id, int tok_id);

    void save_feature_map(string filepath);
    void read_feature_map(string filepath);

    bool training = true;

    void gen_brown_cache(Document *doc);
    void gen_gazetteer_cache(Document *doc);
    unordered_map<int, string> *id2feature;


private:
    void read_good_features(string file);

    unordered_set<string> *good_features;

    void sentence_start(Document *doc, int sen_id, int tok_id);
    void capitalization(Document *doc, int sen_id, int tok_id);
    void forms(Document *doc, int sen_id, int tok_id);
    void word_type(Document *doc, int sen_id, int tok_id);
    void affixes(Document *doc, int sen_id, int tok_id);
    void context_affixes(Sentence *sentence, int idx);
    void previous_tag1(Document *doc, int sen_id, int tok_id);
    void previous_tag2(Document *doc, int sen_id, int tok_id);
    void previous_tag_pattern(Document *doc, int sen_id, int tok_id);
    void previous_tag_context(Document *doc, int sen_id, int tok_id);
    void brown_cluster(Document *doc, int sen_id, int tok_id);
    void gazetteer(Document *doc, int sen_id, int tok_id);
    void hyphen(Document *doc, int sen_id, int tok_id);
    void update_doc_stats(Document *doc, int sen_id, int tok_id);
    void add_dummy_feature(Document *doc, int sen_id, int tok_id);
    void wikifier(Document *doc, int sen_id, int tok_id);

    void add_feature(Token *token, string feature, double value);
    int get_feature_id(string name);

    string normalize_digits(string token);
    unordered_map<string, int> *feature2id;

    int fcnt = 1;

    void init_brown_clusters();
    void init_gazetteers();

    // variables for brown clusters
    vector<unordered_map<string, string> *> *brown_clusters;
    const vector<string> brown_cluster_paths{
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt",
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brownBllipClusters",
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt"
    };
    const int min_word_freq = 3;
    const vector<int> prefix_len{4,6,10};
    void get_prefix(Token *token);

    // variables for gazetteers
    const string gazetteer_list = "/shared/experiments/ctsai12/workspace/illinois-ner/config/gazetteers-list.txt";
    vector<unordered_set<string> *> *gazetteers;
    vector<unordered_set<string> *> *gazetteers_nocase;
    vector<string> *gazetteer_names;

};

#endif //NER_FEATUREEXTRACTOR_H
