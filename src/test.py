import ngram
from nltk.corpus import gutenberg

# takes quite a while to complete

for file in gutenberg.fileids():
    print(file, len(gutenberg.raw(file)))
    bigram = ngram.NGramModel(3, gutenberg.raw(file)[0:100000], True)
    sent = ngram.detokenize(bigram.random_sentence(sentence=[ngram.NGramConstants.B_OF_SENTENCE]))
    print(file, ": ", sent, "\n\n", sep="")