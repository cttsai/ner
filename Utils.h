//
// Created by ctsai12 on 5/30/17.
//

#ifndef NER_UTILS_H
#define NER_UTILS_H

#include "Document.h"
#include <fstream>
#include <sstream>
#include <gzstream.h>
#include "dirent.h"
#include "FeatureExtractor.h"
#include <algorithm>
//#include <linear-poly2.h>
//#include <linear-weight.h>
#include <linear.h>
//#include "liblinear-poly2-2.01/linear.h"

extern unordered_map<string, string> *label2id;
extern unordered_map<int, string> *id2label;
extern vector<string> *types;

vector<Document *> * readColumnFormatFiles(const char *directory);
void clean_cap_words(vector<Document *> *docs);
void bio_to_bilou(vector<Document *> *docs);
void bilou_to_bio(vector<Document *> *docs);
struct problem * build_svm_problem(vector<Document *> *docs, FeatureExtractor *extractor);
model * train_ner(struct problem *prob,int solver, double c, double eps, double gamma, double coef);
void predict_ner(vector<Document *> *docs, FeatureExtractor *extractor, model *model);
double evaluate_phrases(vector<Document *> *docs);
void select_features(struct problem *train_prob, FeatureExtractor *extractor, double c, string out_file);
void free_docs(vector<Document *> *docs);
void free_models(struct model **models);
void writeColumnFormatFiles(vector<Document *> *docs, string dir);

// experimental codes
void select_features1(struct problem *train_prob, FeatureExtractor *extractor, double c, string out_file);

struct problem * build_weighted_svm_problem(vector<Document *> *docs, FeatureExtractor *extractor, double ow);
void clear_features(vector<Document *> *docs);
void print_label_stats(vector<Document *> *docs);

#endif //NER_UTILS_H
