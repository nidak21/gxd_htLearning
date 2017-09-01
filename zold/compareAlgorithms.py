"""
Play with classifying GXD high throughput experiment indexing using
scikit-learn.
Try several different learning algorithms and compare them.

Inputs are text files with Array Express experiment titles and descriptions.
Try to classify them as 'yes' or 'no' (relevant to GXD or not)
"""
# Author: Jim Kadin

#import sys

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, f1_score, \
		confusion_matrix, classification_report

from gxd_htLearningLib import getTrainingSet, getFormatedMetrics

dataset = getTrainingSet()

docs_train, docs_test, y_train, y_test = train_test_split( dataset.data,
						dataset.target, test_size=0.5)

# define vectorizer and set of params to optimize over
# characters instead of word tokens. 1, 2, and 3 word ngrams.
cv = TfidfVectorizer(analyzer='word',ngram_range=(1,3), stop_words='english' )
			# option: strip_accents='ascii' or 'unicode' or 'None'j
vectorizerParams = {
		    'vect__ngram_range': [ (1,1), (1,2), (1,3), (1,4) ],
	            'vect__stop_words':  [ 'english', None ],
		  }

# define classifiers and their params to optimize over
classifiers = [
		('Perceptron'    , Perceptron(),    {} ),
		('LogisticRegression' , LogisticRegression(), {'clf__C': [.1, 1, 10, 100] }),
		('SGDClassifier' , SGDClassifier(loss='hinge', penalty='l2',
					alpha=1e-3, n_iter=5, random_state=42),
					{'clf__n_iter': [5,7,10]} ),
	      ]

MYBETA=4
fScorer = make_scorer(fbeta_score, beta=MYBETA, pos_label=1)

for classifierName, clf, params in classifiers:
    print
    print classifierName
    params.update(vectorizerParams ) # merge vectorizer & classifier params

    gs = GridSearchCV( Pipeline( [('vect', cv), ('clf', clf)] ), params,
                        scoring=fScorer )

    gsout = gs.fit(docs_train, y_train)
    print "Score of best estimator on the left out cross validation set: %f\n" \
		    %   gsout.best_score_,
    print gsout.best_params_

    trainedClf = gs.best_estimator_

    # TASK: Predict the outcome on the testing set
    y_predicted = trainedClf.predict( docs_test )

    print getFormatedMetrics( y_test, y_predicted, "foo", "blah"),

    #print  classification_report(y_test, y_predicted, labels=[1],target_names=["no"]) 
    #print confusion_matrix(y_test,y_predicted, labels=[0,1])


    print "F1:  %5.3f" % f1_score(y_test, y_predicted, pos_label=1)
    print "F%d:  %5.3f" % (2,
		    fbeta_score(y_test, y_predicted, beta=2, pos_label=1) )
    print "F%d:  %5.3f" % (3,
		    fbeta_score(y_test, y_predicted, beta=3, pos_label=1) )
    print "F%d:  %5.3f" % (MYBETA,
		    fbeta_score(y_test, y_predicted, beta=MYBETA, pos_label=1) )
