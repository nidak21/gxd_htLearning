
# coding: utf-8

# In[1]:

BETA=4		# beta for f-score. >1 weighs recall more than precision
		#   when evaluating models
TITLE = 'SGDClassifier & TfidfVectorizer & StandardScaler w/ beta=4'
#---------------------------------------------------
import sys
import time
sys.path.append('..')
import gxd_htLearningLib as gxdLL

import numpy as np 
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

# Load dataset and split into training and test set
dataset=ht.getTrainingSet()
print "Data Directory: "  + ht.getDatadir()

expIDs = ht.getExpIDs(dataset.filenames)

randForSplit = np.random.randint(1000)	# get random seed 0..1000
# randForSplit = 10			# uncomment to get fixed seed

expIDs_train, expIDs_test, docs_train,   docs_test, y_train,      y_test      = train_test_split(expIDs, dataset.data, dataset.target,
		   test_size=0.25, random_state=randForSplit)

print "\n" + ht.getTrainTestSplitReport(dataset.target, y_train, y_test,
					    random_state=randForSplit)
# In[6]:

# Run GridSearch on various parameters
randForClassifier = np.random.randint(1000)	# get random seed 0..1000
# randForClassifier = 10			# uncomment to get fixed seed

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
             'classifier__alpha':[.01],
             'classifier__learning_rate':['invscaling'], # 'constant'
             'classifier__eta0':[.1],
             'classifier__loss':[ 'log' ], #'hinge', 'log','modified_huber'],
             'classifier__penalty':['l1'], # ,'elasticnet'
             'classifier__n_iter':[5],
            }
# Scorer used by GridSearchCV() to rate the pipeline options
scorer = ht.makeFscorer(beta=BETA)

gs = GridSearchCV(pipeline, parameters, scoring=scorer, cv=5,
		    n_jobs=-1, verbose=1)
gs.fit( docs_train, y_train )

gridSearchTime = time.asctime()

bestEstimator   = gs.best_estimator_
bestVectorizer  = bestEstimator.named_steps["vectorizer"]
bestClassifier  = bestEstimator.named_steps["classifier"]

# In[8]:

# Print details of the best estimator (pipeline)
print "### Title: " + TITLE
print gridSearchTime
print "Classifer random_state = %d" % randForClassifier

print ht.getGridSearchReport(gs, parameters)
print ht.getVectorizerReport(bestVectorizer, nFeatures=10)

# Print metrics on predictions for test and training sets
y_predicted_test  = bestEstimator.predict(docs_test)
y_predicted_train = bestEstimator.predict(docs_train)

print ht.getFormatedMetrics("Training Set", y_train, y_predicted_train, BETA) ,
print ht.getFormatedMetrics("Test Set",     y_test, y_predicted_test, BETA) ,

# print false negatives and positives
print ht.getFalsePosNegReport(y_test, y_predicted_test, expIDs_test)

print "high weight features"
print ht.getInterestingFeaturesReport(bestClassifier.coef_,
				bestVectorizer.get_feature_names(), num=20)
print

mglearn.tools.visualize_coefficients(
    bestClassifier.coef_, 
    bestVectorizer.get_feature_names(), n_top_features=20
)
plt.title(TITLE)
