# N-Gram Language Probability Analysis #

Returns the probability of a pattern of word length N followed by a word derived from a given corpus. Parsing done by nltk. NOTE: some special characters throw it out of whack.

```
> python ngram.py
N-Gram Demonstration:

bi-gram analysis ("sentence", "1"):    0.5
bi-gram analysis ("This", "is"):       1.0

tri-gram analysis ("not with", "the"): 0.5
tri-gram analysis ("$ it", "ended"):   0.5

bi-gram sentence analysis (same w/ tri-gram): 0.5
```

## Documentation: ##

### ```NGramModel``` member functions: ###
```Python
__init__(self, n, corpus, normalization):
```
Returns an NGramModel object with a selected n value. ```normalization``` just lowercases everything if it's value is True.


```Python
probability(self, pattern, word)
```
Parses the pattern and matches the word. returns the n-gram probabilty of a phrase occuring as MLE.


```Python
sent_probability(self, sentence, type=NGramConstants.LOGARITHMIC)
```
Returns the probability of a sentence with length >= N + 1 occuring. Default calculation method is logarithmic.


```Python
random_sentence(sentence=[NGramConstants.B_OF_SENTENCE], most_likely=False)
```
Returns a random sentence in token form.

### Other ###

```Python
detokenize(tokens)
```
Returns a string from a set of tokens.

```Python
save_ngram(ngram, path)
```
Saves an ngram object, ```ngram```, to a binary file at ```path```.

```Python
load_ngram(path)
```
Loads an ngram object stored at ```path```.

## Example: ##
```
from ngram import NGramModel

w_bigram = NGramModel(2, "Thing 1, Thing 2, Thing 3", True)
print(w_bigram.probability("thing", "1"))

s_bigram = NGramModel(2, "Let's get something to eat. Let's get something to go.")
print(s_bigram.sent_probability("Let's get something to eat."))
```
```
0.3333333
0.5
```
