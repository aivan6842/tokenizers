#pragma once

#include <utility>
#include <memory>
#include <map>

#include "Tokenizer.hpp"


class BPETokenizer : Tokenizer {

    private:
    bool is_trained;
    uint32_t vocab_size;
    std::map<int, std::vector<int>> vocab;
    std::map<std::pair<int, int>, int> merges;
    int eos_token;
    int bos_token;
    int unk_token;

    std::pair<int, int> get_most_occuring_byte_pair(std::vector<int> const &text_bytes);
    std::vector<int> merge_tokens(std::vector<int> & bytes, std::pair<int, int> byte_pair, int new_token_value);
    std::map<std::pair<int, int>, int> get_token_stats(std::vector<int> const &tokens);

    public:
    BPETokenizer(uint32_t vocab_size, int eos_token = 1000000, int bos_token = 1000001, int unk_token = 1000002);

    void train(std::string const text);
    std::vector<int> encode(std::string const text) override;
    std::string decode(std::vector<int> const tokens) override;
};