
import sys
sys.path.append('..')
import argparse
import textTuningLib as tl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
#-----------------------
def process():
    args = parseCmdLine()
    randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
			'randForSplit'      : args.randForSplit,
			'randForClassifier' : args.randForClassifier,
			} )
    beta=4		# >1 weighs recall more than precision
    pipeline = Pipeline( [
	#('vectorizer', tl.StemmedCountVectorizer(
	('vectorizer', CountVectorizer(
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			#preprocessor=tl.vectorizer_preprocessor,
			#preprocessor=tl.vectorizer_preprocessor_stem,
			),
	),
	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', LogisticRegression(verbose=0, class_weight='balanced',
			random_state=randomSeeds['randForClassifier']) ),
	] )
    parameters={'vectorizer__ngram_range':[(1,1), (1,2)],
		'vectorizer__min_df':[2],
		'vectorizer__max_df':[.98],
		'vectorizer__preprocessor':[tl.vectorizer_preprocessor,
					    tl.vectorizer_preprocessor_stem],
		'classifier__C':[.00001, ],
		'classifier__solver':[ 'liblinear' ],
		'classifier__penalty':['l2'],
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

    parser.add_argument('randForSplit', nargs='?', default=None, type=int,
			help="random seed for test_train_split")

    parser.add_argument('randForClassifier', nargs='?', default=None, type=int,
			help="random seed for classifier")
    return parser.parse_args()
#-----------------------
if __name__ == "__main__": process()
