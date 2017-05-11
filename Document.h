//
// Created by ctsai12 on 5/9/17.
//

#ifndef NER_DOCUMENT_H
#define NER_DOCUMENT_H

#include <vector>
#include <string>

using namespace std;

class Token {

public:

    Token(string s){
        surface = s;
    }

    string surface;
    string label;
};

class Sentence {
public:

    vector<Token> tokens;

};


class Document {
public:

    string id;
    vector<Sentence> sentences;
};


#endif //NER_DOCUMENT_H
