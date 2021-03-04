import ngram
from nltk.corpus import gutenberg

# takes quite a while to complete

for file in gutenberg.fileids():
    print(file, len(gutenberg.raw(file)))
    bigram = ngram.NGramModel(2, gutenberg.raw(file)[0:10000], True)
    sent = ngram.detokenize(bigram.random_sentence(sentence=[ngram.NGramConstants.B_OF_SENTENCE]))
    print(file, ":", sent, ": %", bigram.sent_probability(sent))