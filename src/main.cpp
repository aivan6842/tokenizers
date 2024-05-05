#include <iostream>
#include <string>

#include "BPETokenizer.hpp"


int main() {
    std::string training_text = "Hello my name is Alex! I like to program using python and c++. My main interests are machine learning and sports. I attended the University of Waterloo where I completed my undergraduate degree and now work full time as a machine learning engineer. Outside of work and school I like playing soccer. I have played soccer for nearly 20 years now and plan on playing for as long as I can.";
    std::u32string train(training_text.begin(), training_text.end());
    BPETokenizer tokenizer = BPETokenizer(300);
    tokenizer.train(train);

    std::string inf_t = "Hello! This is my project on tokenizers!";
    std::u32string t(inf_t.begin(), inf_t.end());
    std::cout << tokenizer.decode(tokenizer.encode(t)) << std::endl;
}