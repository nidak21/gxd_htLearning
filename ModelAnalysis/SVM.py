import sys
import textTuningLib as tl
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import SVC
#-----------------------
args = tl.parseCmdLine()
randomSeeds = tl.getRandomSeeds( { 	# None means generate a random seed
		'randForSplit'      : args.randForSplit,
		'randForClassifier' : args.randForClassifier,
		} )
pipeline = Pipeline( [
    ('vectorizer', TfidfVectorizer(
    #('vectorizer', CountVectorizer(
		    strip_accents='unicode', decode_error='replace',
		    token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
		    stop_words="english",
		    #preprocessor=tl.vectorizer_preprocessor,
		    #preprocessor=tl.vectorizer_preprocessor_stem,
		    ),
    ),
    #('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
    ('scaler'    , MaxAbsScaler(copy=True)),
    ('classifier', SVC(class_weight='balanced',
		    random_state=randomSeeds['randForClassifier'],
		    ) ),
    ] )
parameters={'vectorizer__ngram_range':[(1,1) ],
	    'vectorizer__min_df':[.01],
	    'vectorizer__max_df':[.8],
	    #'vectorizer__preprocessor':[tl.vectorizer_preprocessor,
	    #                           tl.vectorizer_preprocessor_stem],
	    'classifier__C':[   .1,  ],
	    'classifier__gamma':[  .1,],  # 'auto' ?
	    'classifier__kernel':[ 'sigmoid', ],
	    #'classifier__degree':[ 1, ],
	    #'classifier__loss':[ 'hinge',  ],
	    #'classifier__penalty':[ 'l2', ],
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
