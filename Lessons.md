### Some Lessons Learned

* My version of sklearn is 0.18.1
* Randomness in training.
    * most is in train_test_split and in the classifier (e.g., SGD)
    * as far as I can tell, there is none caused by the gridsearch CV fold
    partitioning
    * so fix a random_state for train_test_split and the classifier, give
    random state for gridsearch StratifiedKfold, now change the StratifiedKfold
    random state, run again. Identical results.
    * BUT if you change the random state for anything else, you get different 
    training results.
    * this seems very weird.

Model Tuning
    * SGD with 'log' seems hard to get reasonable recall on test set no matter
    what I try.
    * I cannot reproduce Yasmine's results for  SGD with 'log', even using
    the same params she used.
    * I cannot reproduce Yasmine's results for SGD and 'modified_huber'
    * I get a little better results with 'modified_huber', (can get recall
    up to .9), but not great
    * SGD with 'hinge' and 'invscaling' is pretty good. Recall 90-94 range
    and precision in 65-70 range

Good params:
### Best Pipeline Parameters:
classifier__alpha: 10000
classifier__eta0: 1e-05
classifier__learning_rate: 'invscaling'
classifier__loss: 'hinge'
classifier__penalty: 'l2'
vectorizer__max_df: 0.98
vectorizer__min_df: 2
vectorizer__ngram_range: (1, 3)

Also these seem comparable:
### Best Pipeline Parameters:
classifier__alpha: 1000
classifier__eta0: 0.0001
classifier__learning_rate: 'invscaling'
classifier__loss: 'hinge'
classifier__penalty: 'l2'
vectorizer__max_df: 0.98
vectorizer__min_df: 2
vectorizer__ngram_range: (1, 3)

Tried CountVectorizer and MaxAbsScaler with hinge, worked comparably.
(not CountVectorizer with StandardScaler throws int to float warnings)
### Best Pipeline Parameters:
classifier__alpha: 100
classifier__eta0: 1e-05
classifier__learning_rate: 'invscaling'
classifier__loss: 'hinge'
classifier__penalty: 'l2'
vectorizer__max_df: 0.98
vectorizer__min_df: 2
vectorizer__ngram_range: (1, 3)

Tried Binary Vectorizer - not good.

CountVectorizer + MaxAbsScaler is getting 95% recall with:
### Best Pipeline Parameters:
classifier__alpha: 1
classifier__eta0: 0.01
classifier__learning_rate: 'invscaling'
classifier__loss: 'hinge'
classifier__penalty: 'l2'
vectorizer__max_df: 0.98
vectorizer__min_df: 2
vectorizer__ngram_range: (1, 3)

### Stemming is weird
* there seems to be no way to add a "stemming" option to a vectorizer that you 
can turn on and off in grid search like other params.
* you can write an analyzer algorithm that adds stemming, but it doesn't stem the individual words within an n-gram, only the last word in the n-gram (maybe that is what the world wants, but it seems weird to me)
* ON FURTHER THOUGHT - the difference between stemming before or after n-gram'ing is probably negligible 
* see https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
* this gives several approaches. Probably the best is creating a subclass of your favorite vectorizer that that overrides the build_analzer() method so it stems (n-grams in the weird way)
* Still pondering how to stem each word before n-gram'ing - seems like you'd have to add the word splitting and stemming into the preprocessing step
* OR play with specifying the tokenizer in the vectorizer. I don't understand who does what in the analyzer vs. tokenizer (e.g., unicode handling, word splitting, n-gram building...)
