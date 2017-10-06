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
import re
import pickle
import argparse
from ConfigParser import ConfigParser
import gxd_htLearningLib as htLib
import textTuningLib as ppLib	# module holding the preprocessor function

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

DATA_TO_PREDICT	 = cp.get("DEFAULT", "DATA_TO_PREDICT")
DEFAULT_OUTPUT   = "predicted.txt"
ENCODE_ID_PREFIX = cp.get("DEFAULT", "ENCODE_ID_PREFIX")
BLESSED_MODEL	 = cp.get("DEFAULT", "BLESSED_MODEL")
PREPROCESSOR     = cp.get("DEFAULT", "PREPROCESSOR")

CLASS_NAMES      = eval( cp.get("DEFAULT", "CLASS_NAMES") )

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		    description='predict relevance of GXD HT experiments')

    parser.add_argument('-i', '--input', dest='inputFile', action='store', 
	required=False, default=DATA_TO_PREDICT,
    	help='tab-delimited experiment input file')

    parser.add_argument('-o', '--output', dest='outputFile', action='store',
	required=False, default=DEFAULT_OUTPUT,
    	help='tab-delimited output file')

    parser.add_argument('-b', '--blessed', dest='blessedModel', action='store',
	required=False, default=BLESSED_MODEL,
    	help='pickled model file')

#    parser.add_argument('-p', '--preprocessor', dest='preprocessor',
#	action='store', required=False, default=None,
#    	help='preprocessor function name in %s' % TUNINGMODULE)

#    parser.add_argument('--noencode', dest='omitEncode',
#	action='store_const', required=False, default=False, const=True,
#    	help='omit Encode experiments w/ ID prefixes "%s" from the dataset'  \
#						% ENCODE_ID_PREFIX)
#
    args = parser.parse_args()
    return args
#----------------------
  
# Main prog
def main():
    args = parseCmdLine()
    #print args

    with open(args.blessedModel, 'rb') as bp:
	blessedModel = pickle.load(bp)

    if PREPROCESSOR != 'None':
	preprocess = getattr( ppLib, PREPROCESSOR )
    else: preprocess = None

    ip = open(args.inputFile, 'r')

    docs = []		# list of text docs (experiments) to be predicted
    ids = []		# parallel list of experiment ids
    titles = []
    descriptions = []
    expFactors = []
    for expLine in ip.readlines()[1:]:

	expFactorStr, desc, expId, title = expLine.split('\t')
	if htLib.isEncodeExperiment(title):	# need to add omit flag check
            print "Skipping ENCODE experiment: '%s'" % ID
            continue
	doc = htLib.constructDoc( title, desc, expFactorStr)
	if preprocess: doc = preprocess(doc)

	docs.append(doc)
	ids.append(expId)
	titles.append(title)
	descriptions.append(desc)
	expFactors.append(expFactorStr)

    y_predict = blessedModel.predict(docs)
    with  open(args.outputFile, 'w') as op:
	for doc, id, title, desc, expFactorStr, y in \
		zip(docs, ids, titles, descriptions, expFactors, y_predict): 

	    print id
	    op.write('\t'.join( [\
				id,
				CLASS_NAMES[y],
				expFactorStr.strip(),
				title.strip(),
				desc.strip(),
				str(doc).strip(),
				]
			    ) + '\n')
#
main()
