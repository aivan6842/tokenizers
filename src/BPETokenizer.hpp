#pragma once

#include <utility>
#include <memory>
#include <map>
#include <vector>
#include <filesystem>


class BPETokenizer{

    private:
    bool is_trained;
    uint32_t vocab_size;
    std::map<int, std::vector<int> > vocab;
    std::map<std::pair<int, int>, int> merges;
    
    int eos_token;
    int bos_token;
    int unk_token;

    std::pair<int, int> get_most_occuring_byte_pair(std::vector<int> const &text_bytes);
    std::vector<int> merge_tokens(std::vector<int> & bytes, std::pair<int, int> &byte_pair, int new_token_value);
    std::pair<std::map<std::pair<int, int>, int>, std::vector<std::pair<int, int>>> get_token_stats(std::vector<int> const &tokens);

    static std::string vocab_file_ext, merges_file_ext, special_tokens_file_ext;

    public:
    BPETokenizer(uint32_t vocab_size, int eos_token = 1000000, int bos_token = 1000001, int unk_token = 1000002);
    BPETokenizer(std::map<int, std::vector<int> > vocab, std::map<std::pair<int, int>, int> merges,int eos_token = 1000000, int bos_token = 1000001, int unk_token = 1000002);

    void train(std::u32string const &text);
    std::vector<int> encode(std::u32string const &text);
    std::string decode(std::vector<int> const &tokens);
    bool tok_is_trained();
    std::map<int, std::vector<int> > get_vocab();
    void save(const std::string &dir);
    static BPETokenizer from_pretrained(const std::string &dir);
};