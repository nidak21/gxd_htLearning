
# coding: utf-8
import numpy as np 

# set random seeds to get reproducible results
randForSplit = None
randForSplit = 10			# uncomment to get fixed seed
if randForSplit == None:
    randForSplit = np.random.randint(1000)	# get random seed 0..1000

randForClassifier = None
randForClassifier = 10		# uncomment to get fixed seed
if randForClassifier == None: 
    randForClassifier = np.random.randint(1000)	# get random seed 0..1000

# Beta for f-score, used by GridSearchCV to evaluate models
BETA=4		# >1 weighs recall more than precision

#---------------------------------------------------
import sys
import time
sys.path.append('..')
import gxd_htLearningLib as gxdLL

import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# uncomment the next line if we run as a notebook. can try 'auto'
#%matplotlib inline
#get_ipython().magic(u'matplotlib inline')	# not sure what this does
#---------------------------------------------------
ht = gxdLL.GxdHtLearningHelper()

# Print Info on this run
print time.asctime()
print "TrainTestSplit random_state = %d" % randForSplit
print "Classifer random_state = %d" % randForClassifier
print "Beta: %d " % BETA

# Load dataset and split into training and test set
dataset=ht.getTrainingSet()
print "Data Directory: %s\n" % ht.getDatadir()

expIDs = ht.getExpIDs(dataset.filenames)

expIDs_train, expIDs_test, docs_train,   docs_test, y_train,      y_test      = train_test_split(expIDs, dataset.data, dataset.target,
		   test_size=0.25, random_state=randForSplit)

# Run GridSearch on various parameters
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
                     strip_accents='unicode', decode_error='replace',
                     token_pattern=u'(?u)\\b([a-z_]\w+)\\b',
		     stop_words="english") ),
    ('scaler'    , StandardScaler(copy=True, with_mean=False, with_std=True) ),
    ('classifier', SGDClassifier(verbose=0, class_weight='balanced',
    			random_state=randForClassifier) ),
    ])
parameters={ 'vectorizer__ngram_range':[(1,3)],
             'vectorizer__min_df':[2],
             'vectorizer__max_df':[.98],
             'classifier__alpha':[1,10, 100, 1000],
             'classifier__learning_rate':['optimal'],
             'classifier__eta0':[.1],
             'classifier__loss':[ 'log' ], # 'log' = Logistic Regression
             'classifier__penalty':['l2'],
             #'classifier__n_iter':[5],
            }
# Scorer used by GridSearchCV() to rate the pipeline options
scorer = ht.makeFscorer(beta=BETA)

gs = GridSearchCV(pipeline, parameters, scoring=scorer, cv=5,
		    n_jobs=-1, verbose=1)
gs.fit( docs_train, y_train )

bestEstimator   = gs.best_estimator_
bestVectorizer  = bestEstimator.named_steps["vectorizer"]
bestClassifier  = bestEstimator.named_steps["classifier"]

# Get metrics on predictions for test and training sets
y_predicted_train = bestEstimator.predict(docs_train)
y_predicted_test  = bestEstimator.predict(docs_test)

print
print ht.getFormatedMetrics("Training Set", y_train, y_predicted_train, BETA) ,
print ht.getFormatedMetrics("Test Set",     y_test, y_predicted_test, BETA) ,

print ht.getGridSearchReport(gs, parameters)
print ht.getVectorizerReport(bestVectorizer, nFeatures=10)

print ht.getInterestingFeaturesReport(bestClassifier.coef_,
				bestVectorizer.get_feature_names(), num=20)
print ht.getFalsePosNegReport(y_test, y_predicted_test, expIDs_test)
print ht.getTrainTestSplitReport(dataset.target, y_train, y_test)

mglearn.tools.visualize_coefficients(
    bestClassifier.coef_, 
    bestVectorizer.get_feature_names(), n_top_features=20
)
plt.title("Visualize Top Weighted Coefficients")
