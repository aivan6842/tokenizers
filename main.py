import tokenizers_cpp
from min_bpe.basic import BasicTokenizer
import time


vocab_size = 300

text = "Hello my name is Alex! I like to program using python and c++. My main interests are machine learning and sports. I attended the University of Waterloo where I completed my undergraduate degree and now work full time as a machine learning engineer. Outside of work and school I like playing soccer. I have played soccer for nearly 20 years now and plan on playing for as long as I can."

tok = tokenizers_cpp.BPETokenizer(vocab_size)
s = time.time()
tok.train(text)
print(tok.get_vocab())
print(tok.decode(tok.encode(text)) == text)
# print(time.time() - s)

# s = time.time()
# print(tok.decode(tok.encode("Hello! This is my project on tokenizers!")))
# print(time.time() - s)

basic_tok = BasicTokenizer()
s = time.time()
basic_tok.train(text, vocab_size)
print(basic_tok.vocab)
# print(time.time() - s)

# s = time.time()
print(basic_tok.decode(basic_tok.encode(text)) == text)
# print(time.time() - s)

dir = "/home/alex/Desktop/cpp/tokenizers/test_tokenizer"
tok.save(dir)
loaded_tok = tokenizers_cpp.BPETokenizer.from_pretrained(dir)
print(tok.decode(tok.encode(text)) == text)

