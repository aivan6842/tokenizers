#include <iostream>
#include <string>

#include "BPETokenizer.hpp"


int main() {
    std::string training_text = "Hello my name is Alex! I like to program using python and c++. My main interests are machine learning and sports. I attended the University of Waterloo where I completed my undergraduate degree and now work full time as a machine learning engineer. Outside of work and school I like playing soccer. I have played soccer for nearly 20 years now and plan on playing for as long as I can.";
    BPETokenizer tokenizer = BPETokenizer(2000);
    tokenizer.train(training_text);

    std::cout << tokenizer.decode(tokenizer.encode("Hello! This is my project on tokenizers!")) << std::endl;
}