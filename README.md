# tokenizers
Welcome to my implementation of `tokenizers`! My goal for this project it to understand the ins and out of tokenization through building the tokenizers from scratch! Since this primarily a learning project it will likely miss out on a lot of features that are provided by Hugging Face's version of tokenizers but I will implement the core features necessary for tokenization!

I will be updating this repository as I make progress. My goals are to implement the 
- BPETokenizer
- WordPieceTokenizer
- UnigramTokenizer

# Building the project
I have provided a `main.cpp` file where I added a small example of how the tokenizer is used. You can use it as a starting point to run through the code. I have provided a `MakeFile` which builds the `main` executable and all the necessary dependencies. To build the executable simply execute
```make main```. To clean up simply execute ```make clean```
To install the python package simply use the provided `build.sh` script.
