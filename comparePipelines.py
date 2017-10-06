
# quick script to compare some sklearn Pipelines to each other over multiple
#  train_test_splits().
# Computes Fscore, precision, recall over multiple splits and then
#  computes averages across the splits.
# Also tries "voting" across the different Pipelines to see if that fares
#  better (in my usage so far, it doesn't!)
#
# This could/should be cleaned up a lot, command line args or config vars,
#  refactoring, having to manually define the pipelines here is a pain
#  (at least should import the Pipeline definitions), ...
import sys
sys.path.append('..')
import argparse
import textTuningLib as tl
import numpy
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
#-----------------------
goodPipelines = [ \
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
NUMTRIES = 20
DATADIR = '/Users/jak/work/gxd_htLearning/Data/training/expFactors'
TESTSIZE=0.20
RANDOMSEED=None
BETA=4
INDEX_OF_YES=1
#-----------------------
def y_vote( theYs,	# [ [y1's], [y2's], ...] parallel arrays of class assn's
	    ):
    '''
    Assuming each yi is an list of 0 and 1's,
    Return a parallel list that is the "vote" across all the yi's
    Ties default to 0 at this point...
    '''
    # there must be a better way to do this... I wish zip() could take a list
    #  of lists
    numOnes = theYs[0]	# numOnes[i] will be the number of 1's across y's[i]

    for Y in theYs[1:]:
	for i, val in enumerate(Y):
	    numOnes[i] += val

    votes = [ 0 for i in range(len(numOnes)) ]
    threshold = len(theYs)/2

    for i, c in enumerate(numOnes):
	if numOnes[i] > threshold: votes[i] = 1

    return votes
#-----------------------

def process():
    dataSet = load_files( DATADIR )
    pipelineInfo = [ {	'fscores':0,	# metric totals across each pipeline/try
			'precisions': 0,#    for computing avgs
			'recalls': 0, } for i in range(len(goodPipelines)+1) ]
						# +1 for voted predictions
    for tt in range(NUMTRIES):		# for split "try"s
	docs_train, docs_test, y_train, y_test = \
		train_test_split( dataSet.data, dataSet.target,
				test_size=TESTSIZE, random_state=RANDOMSEED)
	predictions = []	# list of predictions this try

	print "Samples Split"
	for i, pl in enumerate(goodPipelines):

	    pl.fit(docs_train, y_train)

	    y_pred = pl.predict(docs_test)
	    predictions.append(y_pred)

	    precision, recall, fscore, support = \
			    precision_recall_fscore_support( \
				y_test, y_pred, BETA,
				pos_label=INDEX_OF_YES, average='binary')
	    pipelineInfo[i]['fscores']    += fscore
	    pipelineInfo[i]['precisions'] += precision
	    pipelineInfo[i]['recalls']    += recall

	    l="Pipeline %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			    % (i, BETA, fscore, precision, recall)
	    print l

	vote_pred = y_vote( predictions )
	precision, recall, fscore, support = \
			precision_recall_fscore_support( \
			    y_test, vote_pred, BETA,
			    pos_label=INDEX_OF_YES, average='binary')
	i = len(goodPipelines)
	pipelineInfo[i]['fscores']    += fscore
	pipelineInfo[i]['precisions'] += precision
	pipelineInfo[i]['recalls']    += recall

	l="Votes    %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			% (i, BETA, fscore, precision, recall)
	print l

    # averages across all the tries
    print
    for i in range(len(goodPipelines)+1):
	avgFscore    = pipelineInfo[i]['fscores'] / NUMTRIES
	avgPrecision = pipelineInfo[i]['precisions'] / NUMTRIES
	avgRecall    = pipelineInfo[i]['recalls'] / NUMTRIES
	l="Average  %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			% (i, BETA, avgFscore, avgPrecision, avgRecall)
	print l
#-----------------------
if __name__ == "__main__": process()
