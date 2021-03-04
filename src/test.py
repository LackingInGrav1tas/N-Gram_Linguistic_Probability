import ngram
from nltk.corpus import gutenberg

# takes quite a while to complete

for file in gutenberg.fileids():
    print("processing", file, " ->", len(gutenberg.raw(file)))
    bigram = ngram.NGramModel(2, gutenberg.raw(file), True)
    sent = ngram.detokenize(bigram.random_sentence())
    print(file, ":", sent, ": %", bigram.sent_probability(sent))