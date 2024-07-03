import tokenizers_cpp
from min_bpe.basic import BasicTokenizer
import time


vocab_size = 300


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
