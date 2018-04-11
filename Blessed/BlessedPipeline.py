
# Definitions of good Pipelines to compare.
# Define a variable "pipelines" that is the list of Pipelines.

import sys
sys.path.append('..')
import numpy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
import sklearnHelperLib as ppLib 
#-----------------------

global pipelines
pipelines = [ \
    Pipeline( [
	('vectorizer', TfidfVectorizer( analyzer='word',
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			max_df=0.8, min_df=.01,
			ngram_range=(1,1),
			),
	),
	('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	#('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', LinearSVC(verbose=0,
			loss='hinge', penalty='l2',
			C=0.0001,
			max_iter=200,
			class_weight='balanced',) ),
	] ),
]
#-----------------------
