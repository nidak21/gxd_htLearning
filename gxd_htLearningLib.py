'''
Common stuff for applying sklearn to gxd High Throughput experiment indexing
'''
import sys
import os
import os.path
from ConfigParser import ConfigParser

from sklearn.datasets import load_files
from sklearn.metrics import make_scorer, fbeta_score,\
			    classification_report, confusion_matrix
#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg", "../config.cfg"])

DATADIR = cp.get("DEFAULT", "DATADIR")

# in the list of labels for evaluating text data for including into GXD...
INDEX_OF_YES = 1
INDEX_OF_NO = 0
LABELS = [ INDEX_OF_YES, INDEX_OF_NO ]
TARGET_NAMES = ['yes', 'no']

class GxdHtLearningHelper (object):

    def __init__(self):
	self.datadir = ''
    #---------------------

    def getTrainingSet(self, datadir=DATADIR):
	self.datadir = datadir
	return load_files(datadir)
    # ---------------------------

    def getDatadir(self):
	return self.datadir
    # ---------------------------

    def getExpIDs(self, filenames):
	'''
	Convert list of filenames into experiment IDs
	'''
	return [ os.path.basename(fn) for fn in filenames ]
    # ---------------------------

    def makeFscorer(self, beta=1):
	'''
	Return an fbeta_score function that scores the 'yes's
	'''
	return make_scorer(fbeta_score, beta=beta, pos_label=INDEX_OF_YES)
    # ---------------------------

    def getTrainTestSplitReport( self, \
	y_all,		# numpy.ndarray w/ true classifications for whole set
	y_train,	# ... for split out training set
	y_test,		# ... for split out test set
	random_state=None # report if not None
	):
	'''
	Report on the sizes and makeup of the training and test sets
	'''
	output = ''
	if random_state != None:
	    output += "TrainTestSplit random_state = %d\n" % random_state

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

    def getGridSearchReport( self, \
	gs,	    # sklearn.model_selection.GridsearchCV that has been .fit()
	parameters  # dict of parameters used in the gridsearch
	):
	output = "### Start GridSearch Report\n"

	output += 'Parameter Options Tried:\n'
	for key in sorted(parameters.keys()):
	    output += "%s:%s\n" % (key, str(parameters[key])) 

	output += "\n"

	output += 'GridSearch Pipeline:\n'
	for stepName, obj in gs.best_estimator_.named_steps.items():
	    output += "%s:\n%s\n\n" % (stepName, obj)

	output += 'Best Pipeline Parameters:\n'
	for pName in sorted(parameters.keys()):
	    output += "%s: %r\n" % ( pName, gs.best_params_[pName] )
	output += "### End GridSearch Report\n"
	return output
    # ---------------------------

    def getVectorizerReport(self, vectorizer, nFeatures=10):
	'''
	Format a report on a fitted vectorizer, return string
	'''
	featureNames = vectorizer.get_feature_names()
	midFeature   = len(featureNames)/2

	output =  "Vectorizer:   Number of Features: %d\n\n" % len(featureNames)

	output += "First %d features: %s\n\n" % (nFeatures,
		    format(featureNames[:nFeatures]) )
	output += "Middle %d features: %s\n\n" % (nFeatures,
		    format(featureNames[ midFeature : midFeature+nFeatures]) )
	output += "Last %d features: %s\n" % (nFeatures,
		    format(featureNames[-nFeatures:]) )
	return output
    # ---------------------------

    def getFalsePosNegReport( self,
	y_true,		# numpy.ndarray w/ true classifications for a test set
	y_predicted,	# ... predicted classifications
	sampleNames,	# list of sample names from test set - parallel 
	num=5		# number of false pos/negs to display
	):
	'''
	Report on the false positives and false negatives in a test set
	'''
	falsePositives = []
	falseNegatives = []

	for trueY, predY, name in zip(y_true, y_predicted, sampleNames):
	    if trueY != predY:
		if predY == 1: falsePositives.append(name)
		else: falseNegatives.append(name)

	output = ''
	output += "False positives: %d\n" % len(falsePositives)
	for name in falsePositives[:num]:
	    output += "%s\n" % name

	output += "\n"
	output += "False negatives: %d\n" % len(falseNegatives)
	for name in falseNegatives[:num]:
	    output += "%s\n" % name

	return output
    # ---------------------------

    def getFormatedMetrics(self, \
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

	output = "%s\n" % title
	output += "%s\n" % (classification_report( \
			    y_true, y_predicted,
			    labels=LABELS[:1], target_names=TARGET_NAMES[:1])
			) # only report on first label: "yes"
	output += "F%d: %5.3f\n\n" % (beta,
				    fbeta_score( y_true, y_predicted, beta,
						 pos_label=INDEX_OF_YES ) )
	output += "%s\n" % self.getFormatedCM(y_true, y_predicted)

	return output
    # ---------------------------

    def getFormatedCM( self,
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

    def getInterestingFeatures( self,
	coefficients,	# list of coeficients
	featureNames,	# list of feature names
	num=10		# number of features of each category to report.
	):
	'''
	Assumes 'coefficients' and 'featureNames' are parallel lists.
	Return 2 lists of pairs, [ (feature, coef), (feature, coef), ... ]
	    features w/ highest positive coefs (descending order)
	    features w/ highest (abs value) negative coefs (desc order abs val)
	    (maybe sometime:
	    features w/ lowest (abs value) coefs (ascending order abs val)
	    )
	'''
	pairList = zip(featureNames, coefficients)

	selCoef = lambda x: x[1]	# select the coefficient in the pair
	sortedFeatures = sorted(pairList, key=selCoef, reverse=True)
	topPos = [ x for x in sortedFeatures[:num] if selCoef(x)>0 ]
	nFeat = len(pairList)
	topNeg = [ x for x in sortedFeatures[nFeat-num : nFeat] if selCoef(x)<0]
	return topPos, topNeg
    # ---------------------------

    def getInterestingFeaturesReport( self,
	coefficients,	# numpy.ndarray of coef's like coef_ from a classifier
	featureNames,	# list of feature names
	num=10		# number of features of each category to report.
	):
	# convert from estimator.coef_ numpy.ndarray to coef list
	coefList = coefficients[0].tolist()

	topPos, topNeg = self.getInterestingFeatures(coefList,
						featureNames, num)
	output = "### Start top positive features\n"
	for f,c in topPos:
	    output += "%+3.2f\t%s\n" % (c,f)

	output += "### End top positive features\n"

	output += "### Start top negative features\n"
	for f,c in topNeg:
	    output += "%+3.2f\t%s\n" % (c,f)

	output += "### End top negative features\n"
	return output
    # ---------------------------
# end class GxdHtHelper
