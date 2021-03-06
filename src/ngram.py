import nltk
import enum
import math
import random
import pickle

class NGramConstants(enum.Enum):
    B_OF_SENTENCE = 1
    E_OF_SENTENCE = 2
    NONE = 3
    LAPLACE = 4
    DECIMAL = 5
    LOGARITHMIC = 6

def detokenize(tokens):
    s = ""
    for token in tokens:
        if (token in [NGramConstants.B_OF_SENTENCE, NGramConstants.E_OF_SENTENCE, ","] or token[0] == "'") and len(s) > 0:
            l = list(s)
            l.pop()
            s = ""
            for c in l:
                s += c
        if token in [NGramConstants.E_OF_SENTENCE]:
            s += "."
        elif token != NGramConstants.B_OF_SENTENCE:
            s += token + " "
    return s

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

def smooth(found, words, total, smoothing, k=1):
    if smoothing == NGramConstants.LAPLACE:
        return ( (found + k)  / ( words + total) )
    else:
        return ( found  / words )

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
            w_tokens.append(NGramConstants.E_OF_SENTENCE)
            w_tokens.insert(0, NGramConstants.B_OF_SENTENCE)
            pre_tokens.append(w_tokens)

        tokens = nltk.flatten(pre_tokens)

        words = Map()
        follow_count = Map()

        for i in range(n, len(tokens)+1):
            trange = tokens[i-n:i]
            if not words.contains(trange):
                words.add(trange, 0)
            words.change(trange, words.get(trange) + 1)
            if i != len(tokens):
                if not follow_count.contains((trange, tokens[i])):
                    follow_count.add((trange, tokens[i]), 0)
                follow_count.change((trange, tokens[i]), follow_count.get((trange, tokens[i])) + 1)
        
        self.word_len = len(words.keys)  
        self._probabilities = Map()

        for i in range(len(follow_count.keys)):
            key = follow_count.keys[i]
            self._probabilities.add( key, ( follow_count.get( key ),  words.get(key[0]) ) )


    def random_sentence(self, sentence=[NGramConstants.B_OF_SENTENCE], smoothing=NGramConstants.NONE, k=1, most_likely=False):
        """Returns a randomly generated sentence."""
        if self.n != len(sentence):
            sentence = [NGramConstants.B_OF_SENTENCE]
            for i in range(1, self.n):
                while True:
                    pair = self._probabilities.keys[random.randrange(len(self._probabilities.keys))]
                    if pair[0][i] == sentence[-1]:
                        sentence.append(pair[1])
                        break
        if not most_likely:
            while sentence[-1] != NGramConstants.E_OF_SENTENCE:
                pair = self._probabilities.keys[random.randrange(len(self._probabilities.keys))]
                if pair[0] == sentence[-self.n:]:
                    sentence.append(pair[1])
        else:
            while True:
                pair = self._probabilities.keys[random.randrange(len(self._probabilities.keys))]
                if pair[0] == sentence[self.n:-1]:
                    sentence.append(pair[1])
                    break
            while sentence[-1] != NGramConstants.E_OF_SENTENCE:
                most = (None, 0)
                for i in range(len(self._probabilities.keys)):
                    key = self._probabilities.keys[i]
                    if key[0] == sentence[-self.n:]:
                        x = self._probabilities.vals[i]
                        prob = smooth(x[0], x[1], self.word_len, smoothing, k)
                        if prob > most[1]:
                            most = (key[1], prob)
                sentence.append(most[0])
        return sentence

    def probability(self, pattern, word, smoothing=NGramConstants.NONE, k=1):
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
            if tokens[i] in [".", "!", "?"]:
                pat.append(NGramConstants.E_OF_SENTENCE)
                if i != len(tokens)-1:
                    pat.append(NGramConstants.B_OF_SENTENCE)
            elif tokens[i] == "$":
                if i != 0:
                    if tokens[i-1] == "\\":
                        pat.pop()
                        pat.append(NGramConstants.B_OF_SENTENCE)
                    else:
                        pat.append(tokens[i])
                else:
                    pat.append(NGramConstants.B_OF_SENTENCE)
            else:
                pat.append(tokens[i])
        p = self._probabilities.get((pat, word))
        return smooth( p[0], p[1], self.word_len,  smoothing, k )

    def sent_probability(self, sentence, type=NGramConstants.LOGARITHMIC, smoothing=NGramConstants.NONE, k=1):
        """Returns the probability of a sentence occuring"""
        if self.normalize:
            sentence = sentence.lower()
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) < self.n+1:
            return 0
        pat = []
        for i in range(len(tokens)):
            if tokens[i] == ".":
                pat.append(NGramConstants.E_OF_SENTENCE)
                if i != len(tokens)-1:
                    pat.append(NGramConstants.B_OF_SENTENCE)
            elif tokens[i] == "$":
                if i != 0:
                    if tokens[i-1] == "\\":
                        pat.pop()
                        pat.append(NGramConstants.B_OF_SENTENCE)
                    else:
                        pat.append(tokens[i])
                else:
                    pat.append(NGramConstants.B_OF_SENTENCE)
            else:
                pat.append(tokens[i])
        if type == NGramConstants.DECIMAL:
            p = 1
            for i in range(self.n, len(pat)):
                x = self._probabilities.get((pat[i-self.n:i], pat[i]))
                p *= smooth(x[0], x[1], self.word_len, smoothing, k)
            return p
        else: # Logarithmic prevents underflow
            p = 0
            for i in range(self.n, len(pat)):
                try:
                    x = self._probabilities.get((pat[i-self.n:i], pat[i]))
                    p += math.log(smooth(x[0], x[1], self.word_len, smoothing, k))
                except ValueError:
                    print("probability error. pat:", pat[i-self.n:i], pat[i])
                    print((pat[i-self.n:i], pat[i]) in self._probabilities.keys)
                    exit()
            return math.exp(p)

def save_ngram(ngram, path):
    """Saves ngram to a binary file at path"""
    with open(path, "wb") as file:
        pickle.dump(ngram, file)
def load_ngram(path):
    """Returns an ngram object stored at path"""
    with open(path, "rb") as file:
        return pickle.load(file)

def main():
    print("N-Gram Demonstration:\n")
    
    bigram = NGramModel(2, "This is sentence 1. This is sentence 2.")
    print('bi-gram analysis ("sentence", "1"):   ' , bigram.probability("sentence", "1"))
    print('bi-gram analysis ("This", "is"):      ' , bigram.probability("This", "is"))

    trigram = NGramModel(3, """It didn't start out here. Not with the scramblers or Rorshach, not with Big Ben or Theseus or the vampires. Most people would say it started with the Fireflies, but they'd be wrong. It ended with those things""", True)
    print('\ntri-gram analysis ("not with", "the"):', trigram.probability("not with", "the"))
    print('tri-gram analysis ("$ it", "ended"):  ', trigram.probability("$ it", "ended"))

    bigram = NGramModel(2, "I want Chinese food. I want English food.", True)
    print("\nbi-gram sentence analysis (same w/ tri-gram):", bigram.sent_probability("$ I want english food."))

if __name__ == "__main__":
    main()