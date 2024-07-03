#include <iostream>
#include <algorithm>
#include <map>
#include <climits>
#include <string>
#include <locale>
#include <codecvt>
#include <fstream>
#include <sstream>

#include "BPETokenizer.hpp"


std::string BPETokenizer::vocab_file_ext = "vocab.txt";
std::string BPETokenizer::merges_file_ext = "merges.txt";
std::string BPETokenizer::special_tokens_file_ext = "special_tokens.txt";

BPETokenizer::BPETokenizer(uint32_t vocab_size, int eos_token, int bos_token, int unk_token) : 
    is_trained{false}, vocab_size{vocab_size}, eos_token{eos_token}, bos_token{bos_token}, unk_token{unk_token}
{
    for (int i = 0; i < 256; i++) {
        this->vocab[i] = std::vector<int>{i};
    }
}

BPETokenizer::BPETokenizer(std::map<int, std::vector<int> > vocab, 
                           std::map<std::pair<int, int>, int> merges,
                           int eos_token, 
                           int bos_token,
                           int unk_token) : 
                           is_trained{true}, 
                           vocab_size{uint32_t(vocab.size())},
                           vocab{vocab},
                           merges{merges},
                           eos_token{eos_token}, 
                           bos_token{bos_token}, 
                           unk_token{unk_token}
{}


std::vector<int> BPETokenizer::merge_tokens(std::vector<int> & bytes, std::pair<int, int>& byte_pair, int new_token_value)
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


std::pair<std::map<std::pair<int, int>, int>, std::vector<std::pair<int, int>>>
BPETokenizer::get_token_stats(std::vector<int> const &tokens) 
{
    std::map<std::pair<int, int>, int> stats{};
    std::vector<std::pair<int, int>> order;

    for (int i = 0; i < int(tokens.size()) - 1; i++) {
        std::pair<int, int> byte_pair = {tokens[i], tokens[i+1]};

        if (stats.find(byte_pair) != stats.end()) {
            stats[byte_pair] += 1;
        } else {
            stats[byte_pair] = 1;
        }
        order.push_back(byte_pair);
    }

    return std::pair(stats, order);
}


void BPETokenizer::train(std::u32string const &text)
{
    if (this->is_trained) { return; }

    // start training
    int num_merges = this->vocab_size - this->vocab.size();
    // std::vector<int> bytes(text.begin(), text.end());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::string utf8_bytes = converter.to_bytes(text);

    std::vector<int> bytes;
    for (unsigned char c : utf8_bytes) {
        bytes.push_back(static_cast<int>(c));
    }

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
    std::pair<std::map<std::pair<int, int>, int>, std::vector<std::pair<int, int>>> stats_and_order = get_token_stats(text_bytes);
    std::map<std::pair<int, int>, int> stats = stats_and_order.first;
    std::vector<std::pair<int, int>> order = stats_and_order.second;

    std::pair<int, int> max_occuring_byte_pair;
    int times = 0;
    for (auto &elem: order) {
        if (stats[elem] > times) {
            times = stats[elem];
            max_occuring_byte_pair = elem;
        }
    }

    return max_occuring_byte_pair;
}


std::vector<int> BPETokenizer::encode(std::u32string const &text)
{
    if (!this->is_trained) {
        throw std::bad_function_call();
    }

    // std::vector<int> tokens = std::vector<int>(text.begin(), text.end());
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::string utf8_bytes = converter.to_bytes(text);

    std::vector<int> tokens;
    for (unsigned char c : utf8_bytes) {
        tokens.push_back(static_cast<int>(c));
    }

    while (tokens.size() >= 2) {
        std::pair<std::map<std::pair<int, int>, int>, std::vector<std::pair<int, int>>> stats_and_order = get_token_stats(tokens);
        std::map<std::pair<int, int>, int> stats = stats_and_order.first;
        std::vector<std::pair<int, int>> order = stats_and_order.second;

        // find token pair that was earliest merged
        std::pair<int, int> earliest_merged_pair;
        int min_merges_val = INT_MAX;
        for (auto &entry: order) {
            if (this->merges.count(entry)) {
                if (this->merges[entry] < min_merges_val) {
                    earliest_merged_pair = entry;
                    min_merges_val = this->merges[entry];
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


std::string BPETokenizer::decode(std::vector<int> const &tokens)
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

    return std::string(repr.begin(), repr.end());
}


bool BPETokenizer::tok_is_trained() {return is_trained; }


std::map<int, std::vector<int> > BPETokenizer::get_vocab() {return vocab; }


void BPETokenizer::save(const std::string &dir) {
    std::filesystem::path dir_path = std::filesystem::path(dir);

    if (!this->is_trained) {
        throw std::bad_function_call();
    }

    bool created_dir = std::filesystem::create_directory(dir_path);

    if (!created_dir) {
        throw std::invalid_argument("Directory already exists.");
    }

    std::ofstream vocab_file, merges_file, special_tokens_file;
    std::string vocab_file_path = (dir_path / BPETokenizer::vocab_file_ext).generic_string();
    std::string merges_file_path = (dir_path / BPETokenizer::merges_file_ext).generic_string();
    std::string special_tokens_file_path = (dir_path / BPETokenizer::special_tokens_file_ext).generic_string();

    // serialize vocab and write to file
    vocab_file.open(vocab_file_path);
    for (auto const &entry : vocab) {
        std::stringstream serialized_entry;
        serialized_entry << entry.first;
        for (int i = 0; i < int(entry.second.size()); i++ ) {
            serialized_entry << " " << entry.second[i];
        }
        vocab_file << serialized_entry.str() << "\n";
    }
    vocab_file.close();

    // serialze merges and write to file
    merges_file.open(merges_file_path);
    for (auto const &entry : merges) {
        std::stringstream serialized_entry;
        serialized_entry << entry.first.first << " " << entry.first.second << " " << entry.second << "\n";
        merges_file << serialized_entry.str();
    }
    merges_file.close();

    // serialize special tokens and write to file
    special_tokens_file.open(special_tokens_file_path);
    special_tokens_file << eos_token << " " << bos_token << " " << unk_token << "\n";
    special_tokens_file.close();
}

BPETokenizer BPETokenizer::from_pretrained(const std::string &dir) {
    std::filesystem::path path = std::filesystem::path(dir);
    
    if (!std::filesystem::is_directory(path)) {
        throw std::invalid_argument("Directory doesn't exist");
    }

    int eos_token, bos_token, unk_token;
    std::map<int, std::vector<int> > vocab;
    std::map<std::pair<int, int>, int> merges;
    
    std::ifstream vocab_file, merges_file, special_tokens_file;
    std::string vocab_file_path = (path / BPETokenizer::vocab_file_ext).generic_string();
    std::string merges_file_path = (path / BPETokenizer::merges_file_ext).generic_string();
    std::string special_tokens_file_path = (path / BPETokenizer::special_tokens_file_ext).generic_string(); 

    // deserialize vocab
    std::string line;
    vocab_file.open(vocab_file_path);
    while (std::getline(vocab_file, line)) {
        std::istringstream iss(line);
        int elem;
        
        // read first elem
        iss >> elem;
        int first = elem;

        std::vector<int> vc;
        while (iss >> elem) {
            vc.push_back(elem);
        }

        vocab[first] = vc;
    }

    // deserialize merges
    line.clear();
    merges_file.open(merges_file_path);
    while (std::getline(merges_file, line)) {
        std::istringstream iss(line);
        int first, second, third;

        iss >> first;
        iss >> second;
        iss >> third;

        merges[std::make_pair(first, second)] = third;
    }

    // deserialize special tokens
    line.clear();
    special_tokens_file.open(special_tokens_file_path);
    std::getline(special_tokens_file, line);
    std::istringstream iss(line);
    iss >> eos_token;
    iss >> bos_token;
    iss >> unk_token;


    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        eos_token=eos_token,
        bos_token=bos_token,
        unk_token=unk_token
    );
}