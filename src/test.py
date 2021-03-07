import ngram
from nltk.corpus import gutenberg

# takes quite a while to complete
bigram = ngram.NGramModel(2, "this is the corpus. this also is the corpus.", True)
print(bigram.probability("this", "is"))
bigram = ngram.NGramModel(2, "this is the corpus. this also is the corpus.", True, ngram.NGramConstants.LAPLACE)
print(bigram.probability("this", "is"))

for file in gutenberg.fileids():
    print(file, len(gutenberg.raw(file)))
    trigram = ngram.NGramModel(3, gutenberg.raw(file)[0:10000], True)
    sent = ngram.detokenize(trigram.random_sentence(sentence=[ngram.NGramConstants.B_OF_SENTENCE]))
    print(file, ": ", sent, "\n\n", sep="")