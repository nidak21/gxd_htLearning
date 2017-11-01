#!/usr/bin/env python2.7 
#
# Script to take a tab-delimited GXD HT experiment file containing
#  un-classified (evaluation status = "Not Evaluated")
# experiments and predict their relevance based on the "Blessed model".
#
# Write out a new tab-delimited file included the predicted relevance
#  ('yes' or 'no')
#
# The input file columns are:
# 	ArrayExpress experiment ID
#	title
#	description
#	Experimental factor terms in one field joined by ' | ' into one
#		text string.
#
# Author: Jim Kadin
#

# standard libs
import sys
sys.path.append('..')
sys.path.append('../..')
import string
import pickle
import argparse
from ConfigParser import ConfigParser
import gxd_htLearningLib as htLib
import sklearnHelperLib as ppLib	# module holding preprocessor function

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

DATA_TO_PREDICT	 = cp.get("DEFAULT", "DATA_TO_PREDICT")
BLESSED_MODEL	 = cp.get("DEFAULT", "BLESSED_MODEL")
PREPROCESSOR     = cp.get("DEFAULT", "PREPROCESSOR")
KEEP_ENCODE      = cp.getboolean("DEFAULT", "KEEP_ENCODE")
DEFAULT_OUTPUT   = "predicted.tsv"
DEFAULT_CONF     = "predictedConfidence.tsv"

CLASS_NAMES      = eval( cp.get("DEFAULT", "CLASS_NAMES") )

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		    description='predict relevance of GXD HT experiments')

    parser.add_argument('-i', '--input', dest='inputFile', action='store', 
	required=False, default=DATA_TO_PREDICT,
    	help='tab-delimited experiment input file. Default: %s' \
				% DATA_TO_PREDICT)

    parser.add_argument('-o', '--output', dest='outputFile', action='store',
	required=False, default=DEFAULT_OUTPUT,
    	help='tab-delimited output file. Default: %s' % DEFAULT_OUTPUT)

    parser.add_argument('-c', '--confidence', dest='confFile', action='store',
	required=False, default=DEFAULT_CONF,
    	help='tab-delimited prediction confidences (output) file. Default: %s'\
							% DEFAULT_CONF)

    parser.add_argument('--encode', dest='keepEncode',
        action='store_const', required=False, default=KEEP_ENCODE, const=True,
        help='keep Encode experiments in the dataset. Default: %s'  \
							% str(KEEP_ENCODE)
	)
    parser.add_argument('--noencode', dest='keepEncode',
        action='store_const', required=False, default=not KEEP_ENCODE,
	const=False,
        help='omit Encode experiments in the dataset. Default: %s'  \
							% str(not KEEP_ENCODE)
	)
    parser.add_argument('-p', '--preprocessor', dest='preprocessor',
        action='store', required=False, default=PREPROCESSOR,
        help='preprocessor function name. Default= %s' % PREPROCESSOR)

    parser.add_argument('-b', '--blessed', dest='blessedModel', action='store',
	required=False, default=BLESSED_MODEL,
    	help='pickled model file. Default: %s' % BLESSED_MODEL)

    args = parser.parse_args()
    return args
#----------------------

args = parseCmdLine()
  
# Main prog
def main():

    with open(args.blessedModel, 'rb') as bp:
	blessedModel = pickle.load(bp)

    if args.preprocessor == 'None':
	preprocess = None
    else: preprocess = getattr( ppLib, args.preprocessor )

    docs = []		# list of text docs (experiments) to be predicted
    sampleNames = []		# parallel list of experiment ids
    titles = []
    descriptions = []
    expFactors = []

    # read tab-delimited experiment file
    print "Reading documents from %s...." % args.inputFile
    ip = open(args.inputFile, 'r')
    for expLine in ip.readlines()[1:]:

	expFactorStr, desc, expId, title = \
				    map(string.strip, expLine.split('\t'))
	if not args.keepEncode and htLib.isEncodeExperiment(title):
            print "Skipping ENCODE experiment: '%s'" % expId
            continue

	doc = htLib.constructDoc( title, desc, expFactorStr)
	if preprocess: doc = preprocess(doc)	# this may add white space

	docs.append(str(doc).strip())
	sampleNames.append(expId)
	titles.append(title)
	descriptions.append(desc)
	expFactors.append(expFactorStr)
    print "...done %d documents" % len(docs)

    # PREDICT!
    y_predicted = blessedModel.predict(docs)

    # write prediction file(s)
    writePredictions(blessedModel,
	    sampleNames, y_predicted, expFactors, titles, descriptions, docs)
    return

# ---------------------------
def writePredictions( estimator,
            sampleNames,
            y_predicted,
            expFactors,
            titles,
            descriptions,
            docs
    ):
    '''
    Write prediction file(s).
    We always write a "full" file that includes the predicted classification, 
        experimental factors, title, description, and the processed document.
    If confidence values are available from the estimator, we'll include
        the confidence values in this full file AND write an abbreviated
        "confidence" file that is easier to run analyses on.
    '''
    if hasattr(estimator, "decision_function"):         # have confidence vals
        confs = estimator.decision_function(docs).tolist()
        absConfs = map(abs, confs)

        # prediction tuples for the full prediction file
        fullPreds = zip(sampleNames, y_predicted, confs, absConfs,
                                    expFactors, titles, descriptions, docs)

        # prediction tuples for the abbreviated confidence prediction file
        confPreds = zip(sampleNames, y_predicted, confs, absConfs)

        selConf = lambda x: x[3]        # select abs confidence value 
        fullPreds = sorted(fullPreds, key=selConf)
        confPreds = sorted(confPreds, key=selConf)

        fullHeader = '\t'.join(["Sample",
                                "Prediction",
                                "Confidence",
                                "Abs value",
                                "Experimental Factors",
                                "Title",
                                "Description",
                                "Processed Document",
				]) + '\n'
        fullTemplate = '\t'.join(["%s", "%d", "%5.3f", "%5.3f",
                                        "%s", "%s", "%s", "%s",]
                                ) + '\n'

        confHeader = '\t'.join(["Sample",
                                "Prediction",
                                "Confidence",
                                "Abs value",
                                ]) + '\n'
        confTemplate = '\t'.join(["%s", "%d", "%5.3f", "%5.3f",]) + '\n'

        # write confidence file
	print "Writing confidence file %s...." % args.confFile
        with open(args.confFile, 'w') as fp:
            fp.write(confHeader)
            for p in confPreds:
                fp.write(confTemplate % p)
	print "...done %d lines written" % len(sampleNames)

    else:                       # no confidence values available
        fullPreds = zip(sampleNames, y_predicted,
                                    expFactors, titles, descriptions, docs)

        fullHeader = '\t'.join(["Sample",
                                "Prediction",
                                "Experimental Factors",
                                "Title",
                                "Description",
                                "Processed Document",
                                ]) + '\n'
        fullTemplate = '\t'.join(["%s", "%d", "%s", "%s", "%s", "%s",]) + '\n'

    # write full predictions file
    print "Writing predictions file %s...." % args.outputFile
    with open(args.outputFile, 'w') as fp:
        fp.write(fullHeader)
        for p in fullPreds:
            fp.write(fullTemplate % p)
    print "...done %d lines written" % len(docs)
    return
# ---------------------------
main()
