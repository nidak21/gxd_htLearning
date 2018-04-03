from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
#-----------------------
global blessedPipeline
blessedPipeline = Pipeline( [
('vectorizer', TfidfVectorizer(
		strip_accents='unicode', decode_error='replace',
		token_pattern=u'(?i)\\b([a-z_]\w+)\\b', stop_words="english",
		ngram_range=(1,2),
		max_df=.7,
		min_df=.05
		),
),
('scaler'    , StandardScaler(copy=True,with_mean=False,with_std=True)),
('classifier', LinearSVC(verbose=0, class_weight='balanced',
		penalty='l2', loss='hinge',
		random_state=926,
		C=0.00001,
		max_iter=200) ),
] )
