
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
    Pipeline( [ #   Test  yes       0.64      0.99      0.77       109
	('vectorizer', TfidfVectorizer( analyzer='word',
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			max_df=0.98, min_df=2,
			ngram_range=(1,2),
			),
	),
	('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	#('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', LinearSVC(verbose=0,
			loss='hinge', penalty='l2',
			C=0.00001,
			max_iter=200,
			class_weight='balanced',) ),
	] ),
# Eliminated the Pipeline at an early stage.
#    Pipeline( [ #   Test  yes       0.69      0.98      0.81       110
#	('vectorizer', TfidfVectorizer( analyzer='word',
#			strip_accents='unicode', decode_error='replace',
#			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
#			stop_words="english",
#			max_df=0.98, min_df=2,
#			ngram_range=(1,1),
#			),
#	),
#	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
#	('scaler'    , MaxAbsScaler(copy=True)),
#	('classifier', SGDClassifier(verbose=0,
#			loss='log', penalty='l2',
#			learning_rate='constant',
#			alpha=0.05, eta0=0.01,
#			class_weight='balanced',) ),
#	] ),
    Pipeline( [ #  Test  yes       0.64      0.97      0.77       109
	('vectorizer', CountVectorizer( analyzer='word',
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			max_df=0.98, min_df=2,
			ngram_range=(1,2),
			),
	),
	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', SGDClassifier(verbose=0,
			loss='hinge', penalty='l2',
			learning_rate='constant',
			alpha=0.001, eta0=0.001,
			class_weight='balanced',) ),
	] ),
    Pipeline( [ #  Test  yes       0.71      0.95      0.81       116
	('vectorizer',
	    TfidfVectorizer(analyzer=u'word', binary=False,
		decode_error='replace',
		dtype=numpy.int64, encoding=u'utf-8', input=u'content',
		lowercase=True, max_df=0.98, max_features=None, min_df=2,
		ngram_range=(1,3), norm=u'l2',preprocessor=None,smooth_idf=True,
		stop_words='english',strip_accents='unicode',sublinear_tf=False,
		token_pattern=u'(?i)\\b([a-z_]\\w+)\\b', tokenizer=None,
		use_idf=True, vocabulary=None)
	),
	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', 
	    SVC(C=0.1, cache_size=200, class_weight='balanced', coef0=0.0,
		decision_function_shape=None, degree=3,gamma=0.1,
		kernel='sigmoid', max_iter=-1, probability=False,shrinking=True,
		tol=0.001, verbose=False,) ),
	] ),
]
#-----------------------
