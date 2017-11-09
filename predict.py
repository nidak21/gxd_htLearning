#!/usr/bin/env python2.7 
#
# Script to take a tab-delimited GXD HT experiment file containing
#  un-classified (evaluation status = "Not Evaluated")
# experiments and predict their relevance based on the "Blessed model".
#
# Write out a tab-delimited predictions file(s)
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
DEFAULT_OUTPUT   = "predictions_unknowns.tsv"
DEFAULT_OUTPUT_LONG  = "predictions_long.tsv"

# if we want to print out class names ("yes", "no") instead of 1, 0,
#    we could use this.
#CLASS_NAMES      = eval( cp.get("DEFAULT", "CLASS_NAMES") )

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		    description='predict relevance of GXD HT experiments')

    parser.add_argument('-i', '--input', dest='inputFile', action='store', 
	required=False, default=DATA_TO_PREDICT,
    	help='tab-delimited experiment input file. Default: %s' \
				% DATA_TO_PREDICT)

    parser.add_argument('-o', '--output', dest='outputFile', action='store',
	required=False, default=DEFAULT_OUTPUT,
    	help='output file name, predictions + confidences. Default: %s'\
							% DEFAULT_OUTPUT)

    parser.add_argument('--long', dest='writeLong',
        action='store_const', required=False, default=False, const=True,
        help='write long output file too. Default: False'
	)

    parser.add_argument('-l', '--longfile', dest='longFile', action='store',
	required=False, default=DEFAULT_OUTPUT_LONG,
    	help='long output file name: predictions + text... . Default: %s' \
						% DEFAULT_OUTPUT_LONG)

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
    print "Predicting...."
    y_predicted = blessedModel.predict(docs)
    print "...done"

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
    We always write the "short" file (typically predictions_unknowns.tsv) that
	contains the predictions (1 or 0) and confidence values if available.
    If "writelong" we will also write a big file that includes the title,
    	descriptions, experimental factors, etc.
    '''

    if hasattr(estimator, "decision_function"):         # have confidence vals
        confs = estimator.decision_function(docs).tolist()
        absConfs = map(abs, confs)

        selConf = lambda x: x[2]        # select confidence value for sorting

        # prediction tuples for the "short" prediction file
        preds = zip(sampleNames, y_predicted, confs, absConfs)
        preds = sorted(preds, key=selConf, reverse=True)

        header = '\t'.join(["Sample",
                                "Prediction",
                                "Confidence",
                                "Abs value",
                                ]) + '\n'
        template = '\t'.join(["%s", "%d", "%5.3f", "%5.3f",]) + '\n'

	if args.writeLong:
	    # prediction tuples for the long prediction file
	    fullPreds = zip(sampleNames, y_predicted, confs, absConfs,
					expFactors, titles, descriptions, docs)
	    fullPreds = sorted(fullPreds, key=selConf, reverse=True)

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
    else:                       # no confidence values available
        selExpID = lambda x: x[0]        # select experiment ID for sorting
        # prediction tuples for the "short" prediction file
        preds = zip(sampleNames, y_predicted)
        preds = sorted(preds, key=selExpID)

        header = '\t'.join(["Sample",
			    "Prediction",
			    ]) + '\n'
        template = '\t'.join(["%s", "%d", ]) + '\n'

	if args.writeLong:
	    # prediction tuples for the long prediction file
	    fullPreds = zip(sampleNames, y_predicted,
					expFactors, titles, descriptions, docs)
	    fullPreds = sorted(fullPreds, key=selExpID)

	    fullHeader = '\t'.join(["Sample",
				    "Prediction",
				    "Experimental Factors",
				    "Title",
				    "Description",
				    "Processed Document",
				    ]) + '\n'
	    fullTemplate = '\t'.join(["%s", "%d", "%s", "%s", "%s", "%s",])+'\n'

    # write "short" file
    print "Writing predictions file %s...." % args.outputFile
    with open(args.outputFile, 'w') as fp:
	fp.write(header)
	for p in preds:
	    fp.write(template % p)
    print "...done %d lines written" % len(sampleNames)

    # write "long" predictions file
    if args.writeLong:
	print "Writing long predictions file %s...." % args.longFile
	with open(args.longFile, 'w') as fp:
	    fp.write(fullHeader)
	    for p in fullPreds:
		fp.write(fullTemplate % p)
	print "...done %d lines written" % len(docs)

    return
# ---------------------------
main()
