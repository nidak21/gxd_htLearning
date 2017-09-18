
# test vectorizer(s) to make sure they are getting the right features
import sys
sys.path.append('..')
import argparse
import textTuningLib as tl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#-----------------------
def process():
    #args = parseCmdLine()

    testdoc = [ \
	" Here..here are some! here are things before URL.",
	"http://foo.com.",
	"in between URLs! http://xyz yuck?",
	"...and..some..Stemmed stemming stem",
	]
    print "Running StemmedCountVectorer: w/ custom, non-stemming preprocessor"
    cv = tl.StemmedCountVectorizer( strip_accents='unicode',
			decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			ngram_range=(1,2),
			preprocessor=tl.vectorizer_preprocessor,
			)
    cv.fit(testdoc)
    print cv.get_feature_names()
    print
    print "Running CountVectorizer: with stemming in preprocessor"
    cv = CountVectorizer( strip_accents='unicode',
			decode_error='replace',
			token_pattern=u'(?i)\\b([a-z_]\w+)\\b',
			stop_words="english",
			ngram_range=(1,2),
			preprocessor=tl.vectorizer_preprocessor_stem,)
    cv.fit(testdoc)
    print cv.get_feature_names()
    return

if __name__ == "__main__":
    process()
