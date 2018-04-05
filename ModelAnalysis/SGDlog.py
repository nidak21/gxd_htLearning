import sys
import textTuningLib as tl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import SGDClassifier
#-----------------------
args = tl.parseCmdLine()
randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
		'randForSplit'      : args.randForSplit,
		'randForClassifier' : args.randForClassifier,
		} )
pipeline = Pipeline( [
    #('vectorizer', tl.StemmedCountVectorizer(
    ('vectorizer', CountVectorizer(
    #('vectorizer', TfidfVectorizer(
		    strip_accents='unicode', decode_error='replace',
		    token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
		    stop_words="english",
		    lowercase=True,
		    #preprocessor=tl.vectorizer_preprocessor,
		    ),
    ),
    #('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
    ('scaler'    , MaxAbsScaler(copy=True)),
    ('classifier', SGDClassifier(verbose=0, class_weight='balanced',
		    random_state=randomSeeds['randForClassifier']) ),
    ] )
parameters={'vectorizer__ngram_range':[(1,2)],
	    'vectorizer__min_df':[.01],
	    'vectorizer__max_df':[.8],
	    #'classifier__alpha':[.00001, .0001,.001,.01, ],
	    'classifier__alpha':[.0001, ],
	    #'classifier__eta0':[.00001, .0001,.001,.01, ],
	    'classifier__eta0':[ .0001, ],
	    'classifier__learning_rate':['invscaling', ],
	    #'classifier__learning_rate':['constant','optimal','invscaling'],
	    'classifier__loss':[ 'log' ],
	    'classifier__penalty':['l2', ],
	    }
p = tl.TextPipelineTuningHelper( pipeline, parameters,
		    trainingDataDir=args.trainingData,
		    testSplit=args.testSplit,
		    gridSearchBeta=args.gridSearchBeta,
		    gridSearchCV=args.gridSearchCV,
		    indexOfYes=args.indexOfYes,
		    randomSeeds=randomSeeds,
		    ).fit()
print p.getReports(wIndex=args.wIndex,
		    tuningIndexFile=args.tuningIndexFile,
		    wPredictions=args.wPredictions,
		    predFilePrefix=args.predFilePrefix,
		    compareBeta=args.compareBeta,
		    verbose=args.verbose,
		    )
