#include <iostream>
#include <algorithm>
#include <map>
#include <climits>

#include "BPETokenizer.hpp"


BPETokenizer::BPETokenizer(uint32_t vocab_size, int eos_token, int bos_token, int unk_token) : 
    is_trained{false}, vocab_size{vocab_size}, eos_token{eos_token}, bos_token{bos_token}, unk_token{unk_token}
{
    for (int i = 0; i < 256; i++) {
        this->vocab[i] = std::vector<int>{i};
    }
}


std::vector<int> BPETokenizer::merge_tokens(std::vector<int> & bytes, std::pair<int, int> byte_pair, int new_token_value)
{
    std::vector<int> new_tokens;
    int i = 0;
    
    while (i < int(bytes.size())) {
        if (i < int(bytes.size()) - 1 && bytes[i] == byte_pair.first && bytes[i+1] == byte_pair.second) {
            new_tokens.push_back(new_token_value);
            i ++;
        } else {
            new_tokens.push_back(bytes[i]);
        }
        i++;
    }

    return new_tokens;
}


std::map<std::pair<int, int>, int> BPETokenizer::get_token_stats(std::vector<int> const &tokens) 
{
    std::map<std::pair<int, int>, int> stats{};

    for (int i = 0; i < int(tokens.size()) - 1; i++) {
        std::pair<int, int> byte_pair = {tokens[i], tokens[i+1]};

        if (stats.find(byte_pair) != stats.end()) {
            stats[byte_pair] += 1;
        } else {
            stats[byte_pair] = 1;
        }
    }

    return stats;
}


void BPETokenizer::train(std::string const text)
{
    if (this->is_trained) { return; }

    // start training
    int num_merges = this->vocab_size - this->vocab.size();
    std::vector<int> bytes(text.begin(), text.end());

    for (int i = 0; i < num_merges; i++) {
        int new_token_val = 256 + i; // 256 base vocab size
        std::pair<int, int> most_occuring_byte_pair = get_most_occuring_byte_pair(bytes);
        bytes = merge_tokens(bytes, most_occuring_byte_pair, new_token_val);

        // add to vocabulary
        std::vector<int> new_token_representation = std::vector<int>(
            this->vocab[most_occuring_byte_pair.first].begin(),
            this->vocab[most_occuring_byte_pair.first].end());
        new_token_representation.insert(
            new_token_representation.end(), 
            this->vocab[most_occuring_byte_pair.second].begin(),
            this->vocab[most_occuring_byte_pair.second].end());
        this->vocab[new_token_val] = new_token_representation;

        // add to merges
        this->merges.emplace(most_occuring_byte_pair, new_token_val);
    }

    this->is_trained = true;
}


std::pair<int, int> BPETokenizer::get_most_occuring_byte_pair(std::vector<int> const &text_bytes)
{
    std::map<std::pair<int, int>, int> stats = get_token_stats(text_bytes);

    std::map<std::pair<int, int>, int>::iterator max_occuring_byte_pair = 
        std::max_element(stats.begin(), stats.end(), 
        [](const auto & a, const auto & b) {return a.second < b.second;});

    return max_occuring_byte_pair->first;
}


std::vector<int> BPETokenizer::encode(std::string const text)
{
    if (!this->is_trained) {
        throw std::bad_function_call();
    }
    std::vector<int> tokens = std::vector<int>(text.begin(), text.end());

    while (tokens.size() >= 2) {
        std::map<std::pair<int, int>, int> stats = get_token_stats(tokens);

        // find token pair that was earliest merged
        std::pair<int, int> earliest_merged_pair;
        int min_merges_val = INT_MAX;
        for (auto &entry: stats) {
            if (this->merges.count(entry.first)) {
                if (this->merges[entry.first] < min_merges_val) {
                    earliest_merged_pair = entry.first;
                    min_merges_val = this->merges[entry.first];
                }
            }
        }
        
        // if no mergable pairs were found
        if (min_merges_val == INT_MAX) { break; }

        int new_token_value = this->merges[earliest_merged_pair];
        tokens = merge_tokens(tokens, earliest_merged_pair, new_token_value);
    }

    tokens.emplace(tokens.begin(), this->bos_token);
    tokens.push_back(this->eos_token);
    return tokens;
}


std::string BPETokenizer::decode(std::vector<int> const tokens)
{
    if (tokens[0] != this->bos_token or tokens[tokens.size() - 1] != this->eos_token) {
        throw std::invalid_argument("Invalid tokenization. Should begin with bos token and end in eos token");
    }

    std::vector<int> repr;

    for (int i = 1; i < int(tokens.size()) - 1; i++) {
        if (this->vocab.count(tokens[i])) {
            std::vector<int> token_bytes = this->vocab[tokens[i]];
            repr.insert(repr.end(), token_bytes.begin(), token_bytes.end());
        } else {
            // unknown token
            std::cerr << "Unknown Token: " << tokens[i] << " at position " << i << std::endl;
            repr.push_back(this->unk_token);
        }
    }

    // null terminator
    repr.push_back('\0');

    return std::string(repr.begin(), repr.end());
}