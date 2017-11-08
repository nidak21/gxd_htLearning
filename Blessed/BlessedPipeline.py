# This is the "blessed" pipeline.
# the one that scored the best during Tuning and comparison to other pipelines
#
# This expects to have preprocessed data (stemmed, lowercase, ...) - details?
# This needs to be trained on sample data and saved/pickled as the blessed model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

global blessedPipeline
blessedPipeline = Pipeline( [
    ('vectorizer', TfidfVectorizer( analyzer='word',
		    strip_accents='unicode', decode_error='replace',
		    token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
		    stop_words="english",
		    max_df=0.98, min_df=2,
		    ngram_range=(1,2),
		    ),
    ),
    ('scaler'    ,StandardScaler(copy=True,with_mean=False,with_std=True)),
    ('classifier', LinearSVC(verbose=0,
		    loss='hinge', penalty='l2',
		    C=0.000001,
		    max_iter=200,
		    class_weight='balanced',) )
    ]
    )
