# N-Gram Language Probability Analysis #

Returns the probability of a pattern of word length N followed by a word derived from a given corpus. Parsing done by nltk.

```
> python ngram.py
N-Gram Demonstration:

bi-gram analysis ("sentence", "1"):    0.5
bi-gram analysis ("This", "is"):       1.0
tri-gram analysis ("not with", "the"): 0.5
```

## Documentation: ##

### ```NGramModel```member functions: ###
```Python
__init__(self, n, corpus, normalization):
```
Returns an NGramModel object with a selected n value. ```normalization``` just lowercases everything if it's value is True.


```Python
probability(self, pattern, word)
```
Parses the pattern and matches the word. returns the n-gram probabilty of a phrase occuring as MLE.

## Example: ##
```
from ngram import NGramModel

bigram = NGramModel(2, "Thing 1, Thing 2, Thing 3", True)
print(bigram.probability("thing", "1"))
```
```
0.3333333
```