import ngram
from nltk.corpus import gutenberg

# takes quite a while to complete
bigram = ngram.NGramModel(2, "this is the corpus. this also is the corpus.", True)
rs = ngram.detokenize(bigram.random_sentence())
print(rs, "-> ", bigram.sent_probability("this also is the corpus.", smoothing=ngram.NGramConstants.LAPLACE))
print(bigram.probability("this", "is", smoothing=ngram.NGramConstants.LAPLACE))

for file in gutenberg.fileids():
    print(file, len(gutenberg.raw(file)))
    trigram = ngram.NGramModel(3, gutenberg.raw(file)[0:10000], True)
    print(trigram.random_sentence(sentence=["I", "am"]))
    sent = ngram.detokenize(trigram.random_sentence(sentence=[ngram.NGramConstants.B_OF_SENTENCE]))
    print(file, ": ", sent, "\n\n", sep="")