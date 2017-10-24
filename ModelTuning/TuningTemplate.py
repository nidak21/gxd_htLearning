import sys
sys.path.extend(['..','../..'])
import textTuningLib as tl
import sklearnHelperLib as hl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
#-----------------------
def process():
    args = tl.parseCmdLine()
    randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
			'randForSplit'      : args.randForSplit,
			'randForClassifier' : args.randForClassifier,
			} )
    pipeline = Pipeline( [
	#('vectorizer', hl.StemmedCountVectorizer(
	('vectorizer', TfidfVectorizer(
			strip_accents='unicode', decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			#preprocessor=hl.vectorizer_preprocessor,
			#preprocessor=hl.vectorizer_preprocessor_stem,
			),
	),
	#('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
	('scaler'    , MaxAbsScaler(copy=True)),
	('classifier', SGDClassifier(verbose=0, class_weight='balanced',
			random_state=randomSeeds['randForClassifier']) ),
	] )
    parameters={'vectorizer__ngram_range':[(1,3)],
		'vectorizer__min_df':[2],
		'vectorizer__max_df':[.98],
                #'vectorizer__preprocessor':[hl.vectorizer_preprocessor,
                #                            hl.vectorizer_preprocessor_stem],
		'classifier__alpha':[1],
		'classifier__learning_rate':['invscaling'],
		'classifier__eta0':[ .01],
		'classifier__loss':[ 'hinge' ],
		'classifier__penalty':['l2'],
		}
    ht = tl.TextPipelineTuningHelper( \
	pipeline, parameters, randomSeeds=randomSeeds, #beta=4, cv=5,
	).fit()
    print ht.getReports(verbose=args.verbose)
#-----------------------
if __name__ == "__main__": process()
