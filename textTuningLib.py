'''
Support for "tuning scripts" for text machine learning projects.

Separate from this module, there are tuning scripts that define Pipelines and
parameter options to try/compare via GridSearchCV.

The idea is that, as much as possible, all code is here for these scripts, and
only the Pipelines and parameters to try are in the tuning scripts.

So the code here is coupled with TuningTemplate.py.

The biggest part of this module is the implementation of various tuning
reports used to analyze the tuning runs.

Assumes:
* Tuning a binary text classification pipeline (maybe this assumption can go?)
* 'Yes' is index 1 of list of classification labels (targets)
* 'No' is index 0 
* The text sample data is in sklearn load_files directory structure
* Topmost directory in that folder is specified in config file
* We are Tuning a Pipeline via GridSearchCV
* The Pipeline has named steps: 'vectorizer', 'classifier' with their
*   obvious meanings (may have other steps too)
* We are scoring GridSearchCV Pipeline parameter runs via an F-Score
*   (beta is a parameter to this library)
* probably other things...
*
If the 'classifier' supports getting weighted coefficients via
classifier.coef_, then the output from here can include a TopFeaturesReport

Convention: trying to use camelCase for all the names here, but
    sklearn typically_uses_names with underscores.
'''
import sys
import time
import re
import string
import os
import os.path
sys.path.append('..')
import argparse
from ConfigParser import ConfigParser

import sklearnHelperLib as skhelper

import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score,\
			    classification_report, confusion_matrix
#-----------------------------------
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg", "../config.cfg"])

TRAINING_DATA   = cp.get("DEFAULT", "TRAINING_DATA")
INDEX_OF_YES    = cp.getint("DEFAULT", "INDEX_OF_YES")
INDEX_OF_NO     = cp.getint("DEFAULT", "INDEX_OF_NO")
GRIDSEARCH_BETA = cp.getint("MODEL_TUNING", "GRIDSEARCH_BETA")
COMPARE_BETA    = cp.getint("MODEL_TUNING", "COMPARE_BETA")
TEST_SPLIT      = cp.getfloat("MODEL_TUNING", "TEST_SPLIT")
GRIDSEARCH_CV   = cp.getint("MODEL_TUNING", "GRIDSEARCH_CV")

# in the list of classifications labels for evaluating text data
LABELS = [ INDEX_OF_YES, INDEX_OF_NO ]
TARGET_NAMES = ['yes', 'no']

# ---------------------------
# Common command line parameter handling for the tuning scripts
# ---------------------------

def parseCmdLine():
    """
    shared among the ModelTuning scripts
    """
    parser = argparse.ArgumentParser( \
    description='Run a tuning experiment script')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='verbose: print longer tuning report')

    parser.add_argument('--rsplit', dest='randForSplit',
			default=None, type=int,
                        help="integer random seed for test_train_split")

    parser.add_argument('--rclassifier', dest='randForClassifier',
			default=None, type=int,
                        help="integer random seed for classifier")

    return parser.parse_args()

# ---------------------------
# Random seed support:
# For various methods, random seeds are used
#   e.g., for train_test_split() the seed is used to decide which samples
#         make it into which set.
# However, we want to record/report the random seeds used so we can
#     reproduce results when desired.
#     So we use these routines to always provide and report a random seed.
#     If a seed is provided, we use it, if not, we generate one here.
#
# getRandomSeeds() takes a dictionary of seeds, and generates random seeds
#     for any key that doesn't already have a numeric seed
# getRandomSeedReport() formats a seed dictionary in a standard way
#     for reporting.
# ---------------------------

def getRandomSeeds( seedDict    # dict: {'seedname' : number or None }
    ):
    '''
    Set a random integer for each key in seedDict that is None
    '''
    for k in seedDict.keys():
        if seedDict[k] == None: seedDict[k] = np.random.randint(1000)

    return seedDict
# ---------------------------

def getRandomSeedReport( seedDict ):
    output = "Random Seeds:\t"
    for k in sorted(seedDict.keys()):
        output += "%s=%d   " % (k, seedDict[k])
    return output

# ---------------------------
# Some basic utilities...
# ---------------------------

def makeFscorer(beta=GRIDSEARCH_BETA):
    '''
    Return an fbeta_score function that scores the 'yes's
    '''
    return make_scorer(fbeta_score, beta=beta, pos_label=INDEX_OF_YES)

# Main class
# ---------------------------

class TextPipelineTuningHelper (object):

    def __init__(self,
	pipeline,
	pipelineParameters,
	beta=GRIDSEARCH_BETA,	# beta=None implies use GRIDSEARCH_BETA
	cv=GRIDSEARCH_CV,	# number of cross validation folds to use
	gsVerbose=1,		# verbose setting for grid search (to stdout)
	testSize=TEST_SPLIT,	# size of the test set from the dataSet
	randomSeeds={'randForSplit':1},	# random seeds. Assume all are not None
	nFeaturesReport=10,
	nFalsePosNegReport=5,
	nTopFeatures=20
	):

	self.pipeline = pipeline
	self.pipelineParameters = pipelineParameters

	if beta == None: self.gridSearchBeta = GRIDSEARCH_BETA
	else: self.gridSearchBeta = beta

	self.gs = GridSearchCV(pipeline,
				pipelineParameters,
				scoring=makeFscorer(beta=self.gridSearchBeta),
				cv=cv,
				verbose=gsVerbose,
				n_jobs=-1,
				)
	self.testSize = testSize
	self.randomSeeds = randomSeeds
	self.randForSplit = randomSeeds['randForSplit']	# required seed

	self.nFeaturesReport = nFeaturesReport
	self.nFalsePosNegReport = nFalsePosNegReport
	self.nTopFeatures = nTopFeatures

	self.time = time.asctime()
	self.readDataSet()
	self.findSampleNames()
    #---------------------

    def getDataDir(self):
	return TRAINING_DATA
    # ---------------------------

    def getDataSet(self):
	return self.dataSet
    # ---------------------------

    def getSampleNames(self):
	return self.sampleNames
    # ---------------------------

    def readDataSet(self):
	self.dataSet = load_files( self.getDataDir() )
    # ---------------------------

    def findSampleNames(self):
	'''
	Convert list of filenames into Samplenames
	'''
	self.sampleNames = [ os.path.basename(fn) for fn \
						    in self.dataSet.filenames ]
    # ---------------------------

    def fit(self):
	'''
	run the GridSearchCV
	'''
	# using _train _test variable names as is the custom in sklearn.
	# "y_" are the correct classifications (labels) for the corresponding
	#   samples

	    # sample names
	    # documents (strings) themselves
	    # correct classifications (labels) for the samples
	self.sampleNames_train, self.sampleNames_test,	\
	self.docs_train,        self.docs_test,	\
	self.y_train,           self.y_test = train_test_split( \
					    self.sampleNames,
					    self.dataSet.data,
					    self.dataSet.target,
					    test_size=self.testSize,
					    random_state=self.randForSplit)

	self.gs.fit( self.docs_train, self.y_train )	# DO THE GRIDSEARCH

	# FIXME: Should implement getter's for these
	self.bestEstimator  = self.gs.best_estimator_
	self.bestVectorizer = self.bestEstimator.named_steps['vectorizer']
	self.bestClassifier = self.bestEstimator.named_steps['classifier']

	# run estimator on both the training set and test set so we can compare
	self.y_predicted_train = self.bestEstimator.predict(self.docs_train)
	self.y_predicted_test  = self.bestEstimator.predict(self.docs_test)

	return self		# customary for fit() methods
    # ---------------------------

    def getFalsePosNeg( self):
	'''
	Return lists of (sample names of) false positives and false negatives in
	    a test set
	'''
	y_true      = self.y_test
	y_predicted = self.y_predicted_test
	sampleNames = self.sampleNames_test

	falsePositives = []
	falseNegatives = []

	for trueY, predY, name in zip(y_true, y_predicted, sampleNames):
	    if trueY != predY:
		if predY == 1: falsePositives.append(name)
		else: falseNegatives.append(name)

	return falsePositives, falseNegatives
    # ---------------------------

    def getReports(self, verbose=True):

	output = getReportStart( self.time, self.gridSearchBeta,
					self.randomSeeds, self.getDataDir() )

	output += getFormatedMetrics("Training Set", self.y_train,
					self.y_predicted_train, COMPARE_BETA)
	output += getFormatedMetrics("Test Set", self.y_test,
					self.y_predicted_test, COMPARE_BETA)
	output += getBestParamsReport(self.gs, self.pipelineParameters)
	output += getGridSearchReport(self.gs, self.pipelineParameters)

	if verbose: 
	    output += getTopFeaturesReport( \
		    getOrderedFeatures(self.bestVectorizer,self.bestClassifier),
		    self.nTopFeatures) 

	    output += getVectorizerReport(self.bestVectorizer,
					    nFeatures=self.nFeaturesReport)

	    falsePos, falseNeg = self.getFalsePosNeg()
	    output += getFalsePosNegReport( falsePos, falseNeg,
						num=self.nFalsePosNegReport)

	    output += getTrainTestSplitReport(self.dataSet.target, self.y_train,
						self.y_test, self.testSize)

	output += getReportEnd()
	return output
# end class TextPipelineTuningHelper
# ---------------------------
# Functions to format output reports
# ---------------------------
SSTART = "### "			# output section start delimiter

def getReportStart( curtime, beta, randomSeeds,dataDir):

    output = SSTART + "Start Time %s\n" % curtime
    output += "Data dir: %s,\tGridSearch Beta: %d\n" % (dataDir, beta)
    output += getRandomSeedReport(randomSeeds)
    output += "\n"
    return output
# ---------------------------

def getReportEnd():
    return SSTART + "End Time %s\n" % (time.asctime())
# ---------------------------

def getTrainTestSplitReport( \
	y_all,
	y_train,
	y_test,
	testSize
	):
    '''
    Report on the sizes and makeup of the training and test sets
    FIXME:  this is very yucky code...
    '''
    output = SSTART + 'Train Test Split Report, test %% = %4.2f\n' % (testSize)
    output += \
    "All Samples: %6d\tTraining Samples: %6d\tTest Samples: %6d\n" \
		    % (len(y_all), len(y_train), len(y_test))
    nYesAll = y_all.tolist().count(1)
    nYesTra = y_train.tolist().count(1)
    nYesTes = y_test.tolist().count(1)
    output += \
    "Yes count:   %6d\tYes count:        %6d\tYes count:    %6d\n" \
		    % (nYesAll, nYesTra, nYesTes)
    output += \
    "No  count:   %6d\tNo  count:        %6d\tNo  count:    %6d\n" \
		    % (y_all.tolist().count(0),
		       y_train.tolist().count(0),
		       y_test.tolist().count(0)  )
    output += \
    "Percent Yes:    %2.2d%%\tPercent Yes:         %2.2d%%\tPercent Yes:     %2.2d%%\n" \
		    % (100 * nYesAll/len(y_all),
		       100 * nYesTra/len(y_train),
		       100 * nYesTes/len(y_test) )
    return output
# ---------------------------

def getBestParamsReport( \
    gs,	    # sklearn.model_selection.GridsearchCV that has been .fit()
    parameters  # dict of parameters used in the gridsearch
    ):
    output = SSTART +'Best Pipeline Parameters:\n'
    for pName in sorted(parameters.keys()):
	output += "%s: %r\n" % ( pName, gs.best_params_[pName] )

    output += "\n"
    return output
# ---------------------------

def getGridSearchReport( \
    gs,	    # sklearn.model_selection.GridsearchCV that has been .fit()
    parameters  # dict of parameters used in the gridsearch
    ):
    output = SSTART + 'GridSearch Pipeline:\n'
    for stepName, obj in gs.best_estimator_.named_steps.items():
	output += "%s:\n%s\n\n" % (stepName, obj)

    output += SSTART + 'Parameter Options Tried:\n'
    for key in sorted(parameters.keys()):
	output += "%s:%s\n" % (key, str(parameters[key])) 

    output += "\n"
    return output
# ---------------------------

def getVectorizerReport(vectorizer, nFeatures=10):
    '''
    Format a report on a fitted vectorizer, return string
    '''
    featureNames = vectorizer.get_feature_names()
    midFeature   = len(featureNames)/2

    output =  SSTART + "Vectorizer:   Number of Features: %d\n" \
    						% len(featureNames)
    output += "First %d features: %s\n\n" % (nFeatures,
		format(featureNames[:nFeatures]) )
    output += "Middle %d features: %s\n\n" % (nFeatures,
		format(featureNames[ midFeature : midFeature+nFeatures]) )
    output += "Last %d features: %s\n\n" % (nFeatures,
		format(featureNames[-nFeatures:]) )
    return output
# ---------------------------

def getFalsePosNegReport( \
    falsePositives,	# list of (sample names) of the falsePositives
    falseNegatives,	# ... 
    num=5		# number of false pos/negs to display
    ):
    '''
    Report on the false positives and false negatives in a test set
    '''
    # method to get false positives and falsenegativess lists

    output = SSTART + "False positives: %d\n" % len(falsePositives)
    for name in falsePositives[:num]:
	output += "%s\n" % name

    output += "\n"
    output += SSTART + "False negatives: %d\n" % len(falseNegatives)
    for name in falseNegatives[:num]:
	output += "%s\n" % name

    output += "\n"
    return output
# ---------------------------

def getFormatedMetrics( \
	title,		# string title
	y_true,		# true category assignments
	y_predicted,	# predicted assignments
	beta    	# for the fbeta_score
	):
    '''
    Return formated metrics report
    y_true and y_predicted are lists of integer category indexes.
    Assumes we are using a fbeta score, not a good thing in the long term
    '''
    # concat title string onto all the target category names so
    #  they are easier to differentiate in multiple reports (and you can
    #  grep for them)
    target_names = [ "%s %s" % (title[:5], x) for x in TARGET_NAMES ]

    output = SSTART + "Metrics: %s\n" % title
    output += "%s\n" % (classification_report( \
			y_true, y_predicted,
			labels=LABELS[:1], target_names=target_names[:1])
		    ) # only report on first label: "yes"
    output += "%s F%d: %5.3f\n\n" % (title[:5],beta,
				fbeta_score( y_true, y_predicted, beta,
					     pos_label=INDEX_OF_YES ) )
    output += "%s\n" % getFormatedCM(y_true, y_predicted)

    return output
# ---------------------------

def getFormatedCM( \
    y_true,	# true category assignments for test set
    y_predicted	# predicted assignments
    ):
    '''
    Return (minorly) formated confusion matrix
    FIXME: this could be greatly improved
    '''
    output = "%s\n%s\n" % ( \
		str( TARGET_NAMES ),
		str( confusion_matrix(y_true, y_predicted, labels=LABELS) ) )
    return output
#  ---------------------------

def getOrderedFeatures( vectorizer,	# fitted vectorizer from a pipeline
			classifier	# trained classifier from a pipeline
    ):
    '''
    Return list of pairs, [ (feature, coef), (feature, coef), ... ]
	ordered from highest coef to lowest.
    Assumes:  vectorizer has get_feature_names() method
    '''
    if not hasattr(classifier, 'coef_'): # not all have coef's
	return []

    coefficients = classifier.coef_[0].tolist()
    featureNames = vectorizer.get_feature_names()
    
    pairList = zip(featureNames, coefficients)

    selCoef = lambda x: x[1]	# select the coefficient in the pair
    return sorted(pairList, key=selCoef, reverse=True)
# ----------------------------

def getTopFeaturesReport(  \
    orderedFeatures,	# features: [ ('feature name', coef), ...]
    num=20,		# number of features w/ highest & lowest coefs to rpt
    ):
    '''
    Return report of the features w/ the highest (positive) and lowest
    (negative) coefficients.
    Assumes num < len(orderedFeatures).
    '''
    if len(orderedFeatures) == 0:		# no coefs
	output =  SSTART + "Top positive features - not available\n"
	output += SSTART + "Top negative features - not available\n"
	output += "\n"
	return output

    topPos = orderedFeatures[:num]
    topNeg = orderedFeatures[len(orderedFeatures)-num:]

    output = SSTART + "Top positive features (%d)\n" % len(topPos)
    for f,c in topPos:
	output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    output += SSTART + "Top negative features (%d)\n" % len(topNeg)
    for f,c in topNeg:
	output += "%+5.4f\t%s\n" % (c,f)
    output += "\n"

    return output
# ---------------------------
