### Some Lessons Learned

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
