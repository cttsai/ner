#include <iostream>
#include <fstream>
#include <sstream>
#include "Document.h"
#include "dirent.h"

using namespace std;

vector<Document> readColumnFormatFiles(const char *directory){

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (directory)) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            if(ent->d_name[0] == '.') continue;
            string file_path = string(directory)+ent->d_name;
            Document doc;
            doc.id = ent->d_name;
            Sentence sentence;
            cout << file_path << endl;
            ifstream infile(file_path);
            string line;
            while (getline(infile, line)) {
                cout << line << endl;
                if(line == "\n" && sentence.tokens.size() > 0){
                    cout << sentence.tokens.size() << endl;
                    doc.sentences.push_back(sentence);
                    continue;
                }
                stringstream ss(line);
                string buf, surface, label;
                int c = 0;
                while(ss >> buf) {
                    if(c == 0)
                        surface = buf;
                    else if(c == 5)
                        label = buf;
                    c++;
                }
                Token t(surface);
                t.label = label;
                sentence.tokens.push_back(t);
            }
            cout << doc.sentences.size() ;
            break;
        }
        closedir (dir);
    }
}

int main() {

    //const char *train_dir = "/shared/corpora/ratinov2/NER/Data/GoldData/Reuters/ColumnFormatDocumentsSplit/Train";
    const char *train_dir = "/shared/corpora/ner/wikifier-features/en/train-camera3/";

    readColumnFormatFiles(train_dir);

    return 0;
}