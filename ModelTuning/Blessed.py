import sys
sys.path.extend(['..','../..'])
import textTuningLib as tl
import sklearnHelperLib as hl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
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
		token_pattern=u'(?i)\\b([a-z_]\w+)\\b', stop_words="english",
		#preprocessor=hl.vectorizer_preprocessor,
		#preprocessor=hl.vectorizer_preprocessor_stem,
		),
),
('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
#('scaler'    , MaxAbsScaler(copy=True)),
('classifier', LinearSVC(verbose=0, class_weight='balanced',
		random_state=randomSeeds['randForClassifier'],
		max_iter=200) ),
] )
parameters={'vectorizer__ngram_range':[(1,2) ],
	'vectorizer__min_df':[2],
	'vectorizer__max_df':[.98],
	#'vectorizer__preprocessor':[hl.vectorizer_preprocessor,
	#                            hl.vectorizer_preprocessor_stem],
	'classifier__C':[    0.00001,  ],
	'classifier__loss':[ 'hinge', ],
	'classifier__penalty':[ 'l2', ],
	}
p = tl.TextPipelineTuningHelper( pipeline, parameters, beta=4, cv=5,
			randomSeeds=randomSeeds,).fit()
print p.getReports(verbose=args.verbose,
			index=args.index, indexFile=args.indexFile)
