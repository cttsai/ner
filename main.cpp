#include <iostream>
#include <fstream>
#include <sstream>
#include "Document.h"
#include "dirent.h"
#include "FeatureExtractor.h"
#include "liblinear-2.11/linear.h"

using namespace std;

vector<Document> readColumnFormatFiles(const char *directory){

    vector<Document> docs;

    struct dirent *ent;
    DIR *dir = opendir (directory);
    while ((ent = readdir (dir)) != NULL) {
        if(ent->d_name[0] == '.') continue; // skip '.' and '..'
        string file_path = string(directory)+ent->d_name;
        Document doc(ent->d_name);
        Sentence sentence;
        cout << file_path << endl;
        ifstream infile(file_path);
        string line;
        while (getline(infile, line)) {
            if(line.empty()){
                if(sentence.tokens.size() > 0) {
                    doc.sentences.push_back(sentence);
                    sentence.tokens.clear();
                }
                continue;
            }
            stringstream ss(line);
            string buf, surface, label;
            int c = 0;
            while(ss >> buf) {
                if(c == 0)
                    label = buf;
                else if(c == 5)
                    surface = buf;
                c++;
            }
            Token t(surface);
            t.label = label;
            sentence.tokens.push_back(t);
        }
        if(sentence.tokens.size()>0)
            doc.sentences.push_back(sentence);
        docs.push_back(doc);
    }
    closedir (dir);
    return docs;
}

unordered_map<string, string> label2id;

void generate_svm_files(vector<Document> docs, vector<Document> test_docs, FeatureExtractor &extractor){

    ofstream trainfile ("train.data");

    for(Document doc: docs){
        for(Sentence sentence: doc.sentences){
            for(int i = 0; i < sentence.tokens.size(); i++){
                extractor.extract(sentence, i);
                Token &token = sentence.tokens.at(i);
                vector<int> features;
                for(int f: token.features)
                    features.push_back(f);
                sort(features.begin(), features.end());
                string label = label2id.find(token.label)->second;
                trainfile << label;
                for(int f: features){
                    trainfile << " "+to_string(f)+":1";
                }
                trainfile << "\n";
            }
        }
    }

    trainfile.close();

    extractor.save_feature_map("feature.map");

//    extractor.read_feature_map("feature.map");
}

void predict_file(vector<Document> test_docs, FeatureExtractor &extractor){

    extractor.read_feature_map("feature.map");

    const char *model_file = "/Users/ctsai12/workspace/ner/liblinear-2.11/train.data.model";

    model *model = load_model(model_file);

    struct feature_node *x;
    int max_nr_attr = 64;
    int nr_feature=get_nr_feature(model);
    int n;
    if(model->bias>=0)
        n=nr_feature+1;
    else
        n=nr_feature;

    for(Document doc: test_docs){
        for(Sentence sentence: doc.sentences){
            for(int idx = 0; idx < sentence.tokens.size(); idx++){
                extractor.extract(sentence, idx);
                Token &token = sentence.tokens.at(idx);
                vector<int> features;
                for(int f: token.features)
                    features.push_back(f);
                sort(features.begin(), features.end());
                string label = label2id.find(token.label)->second;

                int i = 0;

                for(int f: features) {
                    if (i >= max_nr_attr - 2)    // need one more for index = -1
                    {
                        max_nr_attr *= 2;
                        x = (struct feature_node *) realloc(x, max_nr_attr * sizeof(struct feature_node));
                    }
                    x[i].index = f;
                    x[i].value = 1;

                    if(x[i].index <= nr_feature)
                        i++;
                }
                if(model->bias>=0)
                {
                    x[i].index = n;
                    x[i].value = model->bias;
                    i++;
                }
                x[i].index = -1;

                double predict_label = predict(model, x);
            }
        }
    }

}



int main() {
    label2id["B-LOC"] = "0";
    label2id["I-LOC"] = "1";
    label2id["B-ORG"] = "2";
    label2id["I-ORG"] = "3";
    label2id["B-PER"] = "4";
    label2id["I-PER"] = "5";
    label2id["B-MISC"] = "6";
    label2id["I-MISC"] = "7";
    label2id["O"] = "8";

    //const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train";
    const char *train_dir = "/Users/ctsai12/workspace/data/train-camera3/";
    const char *test_dir = "/Users/ctsai12/workspace/data/test-camera3/";

//    const vector<Document> &docs = readColumnFormatFiles(train_dir);



    const vector<Document> &test_docs = readColumnFormatFiles(test_dir);

    FeatureExtractor extractor;
//    generate_svm_files(docs, test_docs, extractor);
    predict_file(test_docs, extractor);

    return 0;
}