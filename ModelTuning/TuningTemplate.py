
import sys
sys.path.append('..')
import textTuningLib as tl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

beta=4		# >1 weighs recall more than precision
randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
		    'randForSplit'      : None, # replace w/ numeric if desired
		    'randForClassifier' : None, # same...
		    } )
pipeline = Pipeline( [
    ('vectorizer', TfidfVectorizer(
		    strip_accents='unicode', decode_error='replace',
		    token_pattern=u'(?u)\\b([a-z_]\w+)\\b',
		    stop_words="english",
		    preprocessor=tl.vectorizer_preprocessor) ),
    ('scaler'    , StandardScaler(copy=True, with_mean=False, with_std=True) ),
    ('classifier', SGDClassifier(verbose=0, class_weight='balanced',
		    random_state=randomSeeds['randForClassifier']) ),
    ] )
parameters={'vectorizer__ngram_range':[(1,3)],
	    'vectorizer__min_df':[2],
	    'vectorizer__max_df':[.98],
	    'classifier__alpha':[.01],
	    'classifier__learning_rate':['invscaling'],
	    'classifier__eta0':[.1],
	    'classifier__loss':[ 'log' ], # 'log' = Logistic Regression
	    'classifier__penalty':['l1'],
	    'classifier__n_iter':[5],
            }
ht = tl.TextPipelineTuningHelper( \
    pipeline, parameters, beta=beta, cv=5, randomSeeds=randomSeeds,
    ).fit()

print ht.getReports()
