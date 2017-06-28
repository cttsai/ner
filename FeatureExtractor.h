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
    }
    bool filter_features = false;

    void extract(Document *doc, int sen_id, int tok_id);

    void save_feature_map(string filepath);
    void read_feature_map(string filepath);

    bool training = true;

    void gen_brown_cache(Document *doc);
    void gen_gazetteer_cache(Document *doc);
    void read_good_features(string file);
    unordered_map<int, string> *id2feature;
    int max_idx = 0;

    bool use_sen = true;
    bool use_cap = true;
    bool use_form = true;
    bool use_wordtype = true;
    bool use_affixe = true;
    bool use_pretag1 = true;
    bool use_pretag2 = true;
    bool use_tagpattern = true;
    bool use_tagcontext = true;
    bool use_brown = true;
    bool use_gazetteer = true;
    bool use_hyphen = true;
    bool use_wikifier = false;

    bool brown_initialized = false;
    bool gazetteer_initialized = false;

    int form_context_size = 2;
    int brown_context_size = 2;
    int hyphen_context_size = 2;
    int gazetteer_context_size = 2;

    void read_good_features1(string file);

    int gf_set = -1;
    vector<unordered_set<string> *> *good_features1 = NULL;
    string gazetteer_list = "/home/ctsai12/CLionProjects/NER/gazetteers-list.txt";
    vector<string> brown_cluster_paths{
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-english-wikitext.case-intact.txt-c1000-freq10-v3.txt",
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brownBllipClusters",
            "/shared/corpora/ratinov2/NER/Data/BrownHierarchicalWordClusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt"
    };
    int min_word_freq = 3;
    vector<int> prefix_len{4,6,10};

private:

    unordered_set<string> *good_features;

    void sentence_start(Document *doc, int sen_id, int tok_id);
    void capitalization(Document *doc, int sen_id, int tok_id);
    void forms(Document *doc, int sen_id, int tok_id);
    void word_type(Document *doc, int sen_id, int tok_id);
    void affixes(Document *doc, int sen_id, int tok_id);
    void previous_tag1(Document *doc, int sen_id, int tok_id);
    void previous_tag2(Document *doc, int sen_id, int tok_id);
    void previous_tag_pattern(Document *doc, int sen_id, int tok_id);
    void previous_tag_context(Document *doc, int sen_id, int tok_id);
    void brown_cluster(Document *doc, int sen_id, int tok_id);
    void gazetteer(Document *doc, int sen_id, int tok_id);
    void hyphen(Document *doc, int sen_id, int tok_id);
    void update_doc_stats(Document *doc, int sen_id, int tok_id);
    void wikifier(Document *doc, int sen_id, int tok_id);

    void context_affixes(Sentence *sentence, int idx);
    void context_ner(Document *doc, int sen_id, int tok_id);
    void prev_context_ner(Document *doc, int sen_id, int tok_id);
    void next_context_ner(Document *doc, int sen_id, int tok_id);
    void add_dummy_feature(Document *doc, int sen_id, int tok_id);

    void add_feature(Token *token, string feature, double value);
    int get_feature_id(string name);

    string normalize_digits(string token);
    unordered_map<string, int> *feature2id;

    int fcnt = 1;

    void init_brown_clusters();
    void init_gazetteers();

    // variables for brown clusters
    vector<unordered_map<string, string> *> *brown_clusters;
    void get_prefix(Token *token);

    // variables for gazetteers
    vector<unordered_set<string> *> *gazetteers;
    vector<unordered_set<string> *> *gazetteers_nocase;
    vector<string> *gazetteer_names;

};

#endif //NER_FEATUREEXTRACTOR_H
