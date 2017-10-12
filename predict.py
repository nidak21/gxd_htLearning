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
  
# Main prog
def main():
    args = parseCmdLine()

    with open(args.blessedModel, 'rb') as bp:
	blessedModel = pickle.load(bp)

    if args.preprocessor == 'None':
	preprocess = None
    else: preprocess = getattr( ppLib, args.preprocessor )

    docs = []		# list of text docs (experiments) to be predicted
    ids = []		# parallel list of experiment ids
    titles = []
    descriptions = []
    expFactors = []

    # read tab-delimited experiment file
    ip = open(args.inputFile, 'r')
    for expLine in ip.readlines()[1:]:

	expFactorStr, desc, expId, title = \
				    map(string.strip, expLine.split('\t'))
	if not args.keepEncode and htLib.isEncodeExperiment(title):
            print "Skipping ENCODE experiment: '%s'" % ID
            continue

	doc = htLib.constructDoc( title, desc, expFactorStr)
	if preprocess: doc = preprocess(doc)

	docs.append(doc)
	ids.append(expId)
	titles.append(title)
	descriptions.append(desc)
	expFactors.append(expFactorStr)

    # PREDICT!
    y_predict = blessedModel.predict(docs)

    # write tab-delimited output file with the predictions
    with  open(args.outputFile, 'w') as op:
	op.write('\t'.join( [\
			    "ID",
			    "Prediction",
			    "Experimental Factors",
			    "Title",
			    "Description",
			    "Processed Document"
			    ]
			) + '\n')
	for doc, id, title, desc, expFactorStr, y in \
		zip(docs, ids, titles, descriptions, expFactors, y_predict): 

	    op.write('\t'.join( [\
				id,
				CLASS_NAMES[y],
				expFactorStr,
				title,
				desc,
				str(doc).strip(),
				]
			    ) + '\n')
    print "%d experiment predictions written to %s" % \
    						(len(docs), args.outputFile)
#
main()
