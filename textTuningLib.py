'''
Stuff for helping to tune sklearn pipelines for text classification

Assumes:
* Tuning a binary text classification pipeline
* 'Yes' is index 1 of list of classification labels (targets)
* 'No' is index 0 
* The text sample data is in sklearn load_files directory structure
* Topmost directory in that folder is specified in config file
* We are Tuning a Pipeline via GridsearchCV
* The Pipeline has named steps: 'vectorizer', 'classifier' with their
*   obvious meanings (may have other steps too)
* The 'classifier' supports getting weighted coefficients vi classifier.coef_
* We are scoring GridSearchCV Pipeline parameter runs via an F-Score
*   (beta is a parameter to this library)
* probably other things...
*
Convention: trying to use camelCase for all the names here, but
    sklearn typically_uses_names with underscores.
'''
import sys
import time
import re
import os
import os.path
sys.path.append('..')
from ConfigParser import ConfigParser

import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score,\
			    classification_report, confusion_matrix
#-----------------------------------
# Config, constants, ...
# ---------------------------
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg", "../config.cfg"])

DATADIR = cp.get("DEFAULT", "DATADIR")

# in the list of classifications labels for evaluating text data
INDEX_OF_YES = 1
INDEX_OF_NO = 0
LABELS = [ INDEX_OF_YES, INDEX_OF_NO ]
TARGET_NAMES = ['yes', 'no']

# ---------------------------
# Some basic utilities...
# ---------------------------
def vectorizer_preprocessor(input):
    '''
    Cleanse document (strings) before they are passed to the vectorizer
       tokenizer.
    '''
    # remove URLs
    urls_re = re.compile("^https?:.+$",re.IGNORECASE)
    output = ''
    for s in input.split():
	if urls_re.match(s) != None:
	    #print "skipping token: '%s'" % s
	    continue
	else:
	    output += "%s " % s
    return output
# ---------------------------

def makeFscorer(beta=1):
    '''
    Return an fbeta_score function that scores the 'yes's
    '''
    return make_scorer(fbeta_score, beta=beta, pos_label=INDEX_OF_YES)

# ---------------------------
# Random seed support:
# For various methods, random seeds are used
#   e.g., for train_test_split() the seed is used to shuffle which samples
#         make it into which set.
# However, often we want to record/report the random seeds used so we can
#     reproduce results when desired.
#
# getRandomSeeds() takes a dictionary of seeds, and generates random seeds
#     for any key that doesn't already have a numeric seed
# getRandomSeedReport() just formats a seed dictionary in a standard way
#     for reporting.
# ---------------------------

def getRandomSeeds( seedDict	# dict: {'seedname' : number or None }
    ):
    '''
    Set a random integer for each key in seedDict that is None
    '''
    for k in seedDict.keys():
	if seedDict[k] == None: seedDict[k] = np.random.randint(1000)

    return seedDict
# ---------------------------

def getRandomSeedReport( seedDict ):
    output = "Random Seeds:\n"
    for k in sorted(seedDict.keys()):
	output += "%s: %d\n" % (k, seedDict[k])

    return output

# ---------------------------
# Main class
# ---------------------------

class TextPipelineTuningHelper (object):

    def __init__(self,
	pipeline,
	pipelineParameters,
	beta=1,
	cv=5,			# number of cross validation folds to use
	gsVerbose=1,		# verbose setting for grid search (to stdout)
	testSize=0.20,		# size of the test set from the dataSet
	randomSeeds={'randForSplit':1},	# random seeds. Assume all are not None
	nFeaturesReport=10,
	nFalsePosNegReport=5,
	nInterestingFeatures=20
	):

	self.pipeline = pipeline
	self.pipelineParameters = pipelineParameters
	self.beta = beta

	self.gs = GridSearchCV(pipeline,
				pipelineParameters,
				scoring=makeFscorer(beta=beta),
				cv=cv,
				verbose=gsVerbose,
				n_jobs=-1,
				)
	self.testSize = testSize
	self.randomSeeds = randomSeeds
	self.randForSplit = randomSeeds['randForSplit']	# required seed

	self.nFeaturesReport = nFeaturesReport
	self.nFalsePosNegReport = nFalsePosNegReport
	self.nInterestingFeatures = nInterestingFeatures

	self.time = time.asctime()
	self.readDataSet()
	self.findSampleNames()
    #---------------------

    def getDataDir(self):
	return DATADIR
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
	Do the work!
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

	# need getter's for these
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

    def getInterestingFeatures( self) :
	'''
	Return 2 lists of pairs, [ (feature, coef), (feature, coef), ... ]
	    features w/ highest positive coefs (descending order)
	    features w/ highest (abs value) negative coefs (desc order abs val)
	    JIM: maybe sometime: features w/ lowest (abs value) coefs
		(ascending order abs val)
	    JIM: should convert this so it simply returns sorted list
	    	of all (feature, coef) pairs. let the caller decide how
		they want to access this list
	'''
	coefficients = self.bestClassifier.coef_[0].tolist()
	featureNames = self.bestVectorizer.get_feature_names()
	num          = self.nInterestingFeatures
	
	pairList = zip(featureNames, coefficients)

	selCoef = lambda x: x[1]	# select the coefficient in the pair
	sortedFeatures = sorted(pairList, key=selCoef, reverse=True)

	topPos = [ x for x in sortedFeatures[:num] if selCoef(x)>0 ]

	nFeat = len(pairList)
	topNeg = [ x for x in sortedFeatures[nFeat-num : nFeat] if selCoef(x)<0]

	return topPos, topNeg
    # ---------------------------

    def getReports(self, verbose=True):

	output = getReportStart( self.time, self.beta, self.randomSeeds )

	output += getFormatedMetrics("Training Set", self.y_train,
					self.y_predicted_train, self.beta)
	output += getFormatedMetrics("Test Set", self.y_test,
					self.y_predicted_test, self.beta)
	output += getBestParamsReport(self.gs, self.pipelineParameters)

	if not verbose: return output

	topPos, topNeg = self.getInterestingFeatures()
	output += getInterestingFeaturesReport(topPos,topNeg) 

	output += getGridSearchReport(self.gs, self.pipelineParameters)
	output += getVectorizerReport(self.bestVectorizer,
					nFeatures=self.nFeaturesReport)

	falsePos, falseNeg = self.getFalsePosNeg()
	output += getFalsePosNegReport( falsePos, falseNeg,
					    num=self.nFalsePosNegReport)

	output += getTrainTestSplitReport(self.dataSet.target, self.y_train,
					    self.y_test, self.testSize)

	return output
    # ---------------------------
# end class TextPipelineTuningHelper

# ---------------------------
# Functions to format output reports
# ---------------------------
SSTART = "### "			# output section start delimiter

def getReportStart( curtime, beta, randomSeeds):

    output = SSTART + "Tuning Report %s\nBeta: %d\n" % (curtime, beta)
    output += getRandomSeedReport(randomSeeds)
    output += "\n"
    return output
# ---------------------------

def getTrainTestSplitReport( \
	y_all,
	y_train,
	y_test,
	testSize
	):
    '''
    Report on the sizes and makeup of the training and test sets
    JIM:  this is very yucky...
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

    output = SSTART + "Metrics: %s\n" % title
    output += "%s\n" % (classification_report( \
			y_true, y_predicted,
			labels=LABELS[:1], target_names=TARGET_NAMES[:1])
		    ) # only report on first label: "yes"
    output += "F%d: %5.3f\n\n" % (beta,
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
    '''
    output = "%s\n%s\n" % ( \
		str( TARGET_NAMES ),
		str( confusion_matrix(y_true, y_predicted, labels=LABELS) ) )
    return output
#  ---------------------------

def getInterestingFeaturesReport(  \
    topPos,	# top weighted positive features: [ ('feature name', coef), ...]
    topNeg	# ... for negative weighted features
    ):
    output = SSTART + "Top positive features (%d)\n" % len(topPos)
    for f,c in topPos:
	output += "%+3.2f\t%s\n" % (c,f)
    output += "\n"

    output += SSTART + "Top negative features (%d)\n" % len(topNeg)
    for f,c in topNeg:
	output += "%+3.2f\t%s\n" % (c,f)
    output += "\n"

    return output
# ---------------------------
