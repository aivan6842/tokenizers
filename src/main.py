import tokenizers_cpp

print(tokenizers_cpp.__doc__)

text ="Hello my name is Alex! I like to program using python and c++. My main interests are machine learning and sports. I attended the University of Waterloo where I completed my undergraduate degree and now work full time as a machine learning engineer. Outside of work and school I like playing soccer. I have played soccer for nearly 20 years now and plan on playing for as long as I can."
tok = tokenizers_cpp.BPETokenizer(300)
print(tok.tok_is_trained())
print(tok.get_vocab())
tok.train(text)
print(tok.tok_is_trained())
print(tok.get_vocab())