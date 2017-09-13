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

No "no"s called, all weights are zero
Classifer random_state = 161
TrainTestSplit random_state = 78


Unicode error
Classifer random_state = 220
TrainTestSplit random_state = 335

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
