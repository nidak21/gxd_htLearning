#!/usr/bin/env python2.7 
#
# populateTrainingData.py
#
# Script to take a tab-delimited GXD HT experiment file containing training
# data and split it into individual experiment files, one for each experiment.
#
# Files (experiments) get shoved into two directories, /data/no and /data/yes,
#       that sklearn.datasets.load_files() can read easily.
#
# Expermiments w/ eval state = No and Yes go into their respective directories.
#
# Each experiment file contains the title and description fields concatenated
#       into one text string.
# The file names are the experiment IDs
#
# The input file structure is 4 columns, tab-delimited:
#       title  description, experimental factors, exp ID, eval state
# experimental factors is a string of factor terms separated by '|'
#
# Author: Jim Kadin
#

# standard libs
import sys
sys.path.append('..')
sys.path.append('../..')
import string
import os
import argparse
from ConfigParser import ConfigParser
import gxd_htLearningLib as htLib
import sklearnHelperLib as ppLib	# module holding preprocessor function

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

TRAINING_DATA = cp.get("DEFAULT", "TRAINING_DATA")
PREPROCESSOR  = cp.get("DEFAULT", "PREPROCESSOR")
KEEP_ENCODE   = cp.getboolean("DEFAULT", "KEEP_ENCODE")

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Splits GXD HT training data into sklearn directory structure')

    parser.add_argument('inputFile', action='store', 
    	help='tab-delimited input file of training data')

    parser.add_argument('-o', '--outputDir', dest='outputDir', action='store',
	required=False, default=TRAINING_DATA,
    	help='parent dir where /no and /yes exist. Default=%s' % TRAINING_DATA)

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

    args = parser.parse_args()
    return args
#----------------------

# Main prog
def main():
    args = parseCmdLine()
    #print args

    if args.preprocessor == 'None':
	preprocess = None
    else: preprocess = getattr( ppLib, args.preprocessor )

    # for now assume directories exist
    # could create directories instead.

    counts = { 'yes':0, 'no':0, 'encode':0}
    fp = open( args.inputFile, 'r')
    for line in fp.readlines()[1:]:

	title, expFactorStr, desc, ID, yesNo = \
					map(string.strip, line.split('\t'))

	if not args.keepEncode and htLib.isEncodeExperiment(title):
	    print "Skipping ENCODE experiment: '%s'" % ID
	    counts['encode'] += 1
	    continue
	
	yesNo = yesNo.lower()
	counts[yesNo] += 1

	filename = os.sep.join( [ args.outputDir, yesNo, ID ] )

	with open(filename, 'w') as newFile:

	    doc = htLib.constructDoc( title, desc, expFactorStr)

	    if preprocess: doc = preprocess(doc)
	    newFile.write( doc )

    numFiles = counts['yes'] + counts['no']
    print "%d files written to %s" % (numFiles, args.outputDir)
    print "%d yes, %d no" % (counts['yes'], counts['no'])
    print "%d Encode files skipped" % counts['encode']
#
main()
