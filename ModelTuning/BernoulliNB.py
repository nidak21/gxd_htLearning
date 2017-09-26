
import sys
sys.path.append('..')
import argparse
import textTuningLib as tl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
#from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
#-----------------------
def process():
    args = parseCmdLine()
    randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
			'randForSplit'      : args.randForSplit,
			'randForClassifier' : args.randForClassifier,
			} )
    beta=2		# >1 weighs recall more than precision
    pipeline = Pipeline( [
	#('vectorizer', tl.StemmedCountVectorizer(
	#('vectorizer', TfidfVectorizer(
	('vectorizer', CountVectorizer(
			binary=True,		# Bernoulli wants binary
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			lowercase=True,
			#preprocessor=tl.vectorizer_preprocessor,
			),
	),
	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	#('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', BernoulliNB() ),
	] )
    parameters={'vectorizer__ngram_range':[(1,1),(1,2), (1,3)],
		'vectorizer__min_df':[2],
		'vectorizer__max_df':[.98],
		#'classifier__alpha':[.00001, .0001,.001,.01, ],
		'classifier__alpha':[.01, .1, 1, 10 ],
		#'classifier__eta0':[.00001, .0001, .001, ],
		#'classifier__eta0':[ .0001, ],
		#'classifier__learning_rate':[ 'constant' , 'optimal'],
		#'classifier__learning_rate':['constant','optimal','invscaling'],
		#'classifier__loss':[ 'modified_huber' ],
		#'classifier__penalty':['l2', ],
		}
    ht = tl.TextPipelineTuningHelper( \
	pipeline, parameters, beta=beta, cv=5, randomSeeds=randomSeeds,
	).fit()
    print ht.getReports(verbose=args.verbose)
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
