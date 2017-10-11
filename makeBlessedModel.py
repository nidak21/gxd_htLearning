
# Take the BlessedPipeline, train it, and pickle it as the "BlessedModel"
#
# This could/should be cleaned up a lot, command line args or config vars,
#  refactoring,
import sys
sys.path.append('..')
from ConfigParser import ConfigParser
import argparse
import pickle

import textTuningLib as tl
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from BlessedPipeline import blessedPipeline
#-----------------------
cp = ConfigParser()
cp.optionxform = str	# makekeys case sensitive
cp.read(["config.cfg", "../config.cfg"])

TRAINING_DATA = cp.get("DEFAULT", "TRAINING_DATA")
PICKLE_FILE   = cp.get("DEFAULT", "BLESSED_MODEL")
NUM_TOP_FEATURES=20	# number of highly weighted features to report
#-----------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Train model and pickle it.')

    parser.add_argument('-p', '--pickle', dest='pickleFile',
			default=PICKLE_FILE,
                        help='pickle file to write to. Default: "%s"' \
				% PICKLE_FILE)

    parser.add_argument('-t', '--training', dest='trainingDataDir',
			default=TRAINING_DATA,
                        help='where the training set lives. Default: "%s"' \
				% TRAINING_DATA)

    return parser.parse_args()
#-----------------------

def process():
    args = parseCmdLine()
    print "Loading training data from '%s'" % args.trainingDataDir

    dataSet = load_files( args.trainingDataDir )
    blessedPipeline.fit(dataSet.data, dataSet.target)	# train on all samples

    with open(args.pickleFile, 'wb') as fp:
	pickle.dump(blessedPipeline, fp)

    print "Trained model written to '%s'" % args.pickleFile

    # print features report
    vectorizer = blessedPipeline.named_steps['vectorizer']
    classifier = blessedPipeline.named_steps['classifier']
    orderedFeatures = tl.getOrderedFeatures(vectorizer,classifier)
    print tl.getTopFeaturesReport( orderedFeatures, NUM_TOP_FEATURES)
#-----------------------
if __name__ == "__main__": process()
