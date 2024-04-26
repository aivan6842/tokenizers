#pragma once

#include <vector>
#include <string>


class Tokenizer {

    public:
    virtual std::vector<int> encode(std::string text) = 0;
    virtual std::string decode(std::vector<int> tokens) = 0;
};