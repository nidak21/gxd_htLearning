
import sys
sys.path.append('..')
import time
import argparse
import textTuningLib as tl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

#import numpy as np
#import scipy as sp
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

DATADIR = "/Users/jak/work/gxd_htLearning/Data/Data_expFactors"
#-----------------------
# Naive_bayes Gaussian:
#  Doesn't take multiple classifier params to tune.
#  Also requires non-sparse feature matrix, so we use .toarray()
#  (haven't figured out how to make that conversion within a Pipeline)
#  So no GridSearchCV()
#-----------------------

def process():
    args = parseCmdLine()
    randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
			'randForSplit'      : args.randForSplit,
			'randForClassifier' : args.randForClassifier,
			} )
    beta=4		# >1 weighs recall more than precision

    dataSet = load_files(DATADIR)
    startTime = time.asctime()

    docs_train, docs_test, y_train, y_test = train_test_split(\
				dataSet.data, dataSet.target, test_size=0.2)
    vect = CountVectorizer( strip_accents='unicode', decode_error='replace',
    #vect = TfidfVectorizer( strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			lowercase=True,
			min_df=2, max_df=0.98,
			ngram_range=(1,3),
			#dtype=np.int64,
			#dtype=sp.sparse.csr_matrix,
			#dtype=np.matrixlib.defmatrix.matrix,
			#preprocessor=tl.vectorizer_preprocessor,
			)
    scaler = MaxAbsScaler(copy=True)
    scaler = None
    #scaler = StandardScaler(copy=True,with_mean=False,with_std=True)

    x_train = vect.fit_transform(docs_train)
    #x_train = scaler.fit_transform( x_train )

    classifier =  GaussianNB().fit(x_train.toarray(),y_train)

    y_predicted_train = classifier.predict(x_train.toarray())

    x_test = vect.transform(docs_test)
    #x_test = scaler.transform(x_test)
    y_predicted_test  = classifier.predict(x_test.toarray())

    print tl.getReportStart(startTime, beta, randomSeeds, DATADIR)
    print tl.getFormatedMetrics("Train", y_train, y_predicted_train, beta) 
    print tl.getFormatedMetrics("Test",  y_test,  y_predicted_test, beta) 
    print vect
    print scaler
    print classifier
    print
    print tl.getVectorizerReport( vect )
    print tl.getTrainTestSplitReport( dataSet.target, y_train, y_test, 0.2)
    print tl.getReportEnd()
    return
#-----------------------
def parseCmdLine():
    """
    Usage: scriptname [-v] [ randForSplit [ randForClassifier] ]
    """
    parser = argparse.ArgumentParser( \
    description='Run a tuning experiment, log to tuning.log')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
			help='verbose: print longer tuning report')

    parser.add_argument('randForClassifier', nargs='?', default=None, type=int,
			help="random seed for classifier")

    parser.add_argument('randForSplit', nargs='?', default=None, type=int,
			help="random seed for test_train_split")
    return parser.parse_args()
#-----------------------
if __name__ == "__main__": process()
