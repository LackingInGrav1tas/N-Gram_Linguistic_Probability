import nltk
import enum

class Constants(enum.Enum):
    B_OF_SENTENCE = 1
    E_OF_SENTENCE = 2

class NGramError(Exception):
    pass

class Map:
    def __init__(self):
        self.keys = []
        self.vals = []

    def add(self, k, v):
        self.vals.append(v)
        self.keys.append(k)
    
    def contains(self, k):
        return k in self.keys

    def get(self, k, d=0):
        if self.contains(k):
            return self.vals[self.keys.index(k)]
        else: return d

    def change(self, k, v):
        if self.contains(k):
            self.vals[self.keys.index(k)] = v

class NGramModel:
    def __init__(self, n, corpus, normalize=False):
        """Initializes an n-gram. The param n determines the pattern length"""
        self.normalize = normalize
        if normalize:
            corpus = corpus.lower()
        n -= 1
        self.n = n
        s_tokens = nltk.sent_tokenize(corpus)
        pre_tokens = []
        for sentence in s_tokens:
            w_tokens = nltk.word_tokenize(sentence)[0:-1]
            w_tokens.append(Constants.E_OF_SENTENCE)
            w_tokens.insert(0, Constants.B_OF_SENTENCE)
            pre_tokens.append(w_tokens)

        # flattening
        tokens = []
        for sentence in pre_tokens:
            for token in sentence:
                tokens.append(token)

        words = Map()

        for i in range(n, len(tokens)+1):
            trange = tokens[i-n:i]
            if not words.contains(trange):
                words.add(trange, 0)
            words.change(trange, words.get(trange) + 1)

        follow_count = Map()

        for i in range(n, len(tokens)):
            trange = tokens[i-n:i]
            if not follow_count.contains((trange, tokens[i])):
                follow_count.add((trange, tokens[i]), 0)
            follow_count.change((trange, tokens[i]), follow_count.get((trange, tokens[i])) + 1)
        
        self._probabilities = Map()

        for i in range(len(follow_count.keys)):
            key = follow_count.keys[i]
            self._probabilities.add( key, follow_count.get( key )  / words.get(key[0]) )

    def probability(self, pattern, word):
        """"Analyzes how many times a pattern, with n words, occurs followed by specified word"""
        if self.normalize:
            pattern = pattern.lower()
            word = word.lower()
        tokens = nltk.word_tokenize(pattern)
        if len(tokens) != self.n:
            print("expected", self.n, "words.")
            raise NGramError
        pat = []
        for i in range(len(tokens)):
            if tokens[i] == ".":
                pat.append(Constants.E_OF_SENTENCE)
                if i != len(tokens)-1:
                    pat.append(Constants.B_OF_SENTENCE)
            else:
                pat.append(tokens[i])
        return self._probabilities.get((pat, word))


def main():
    
    print("N-Gram Demonstration:\n")
    
    bigram = NGramModel(2, "This is sentence 1. This is sentence 2.")
    print('bi-gram analysis ("sentence", "1"):   ' , bigram.probability("sentence", "1"))
    print('bi-gram analysis ("This", "is"):      ' , bigram.probability("This", "is"))

    trigram = NGramModel(3, """It didn't start out here. Not with the scramblers or Rorshach, not with Big Ben or Theseus or the vampires. Most people would say it started with the Fireflies, but they'd be wrong. It ended with those things""", True)
    print('tri-gram analysis ("not with", "the"):', trigram.probability("not with", "the"))

if __name__ == "__main__":
    main()