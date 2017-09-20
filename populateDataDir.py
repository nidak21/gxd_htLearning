#!/usr/bin/env python2.7 
#
# createDataDir.py
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
#       title  description exp ID, eval state
#
# Author: Jim Kadin
#

# standard libs
import sys
sys.path.append('..')
sys.path.append('../..')
import string
import os
import textTuningLib
TUNINGMODULE = 'textTuningLib'
#import itertools
#import time
#import types
import argparse
from ConfigParser import ConfigParser

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

DATADIR	= cp.get("DEFAULT", "DATADIR")
ENCODE_ID_PREFIX = "ENCSR"	# Prefix of Encode IDs (not ArrayExpress IDs)
				#   that appear in the title of many Encode
				#   titles.
				# The titles/desc's of these exp's are not
				#  informative.
				# We provide an option to exclude these from
				#  the sample set.
def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Splits GXD HT training data into sklearn directory structure')

    parser.add_argument('inputFile', action='store', 
    	help='tab-delimited input file of training data')

    parser.add_argument('-p', '--preprocessor', dest='preprocessor',
	action='store', required=False, default=None,
    	help='preprocessor function name in %s' % TUNINGMODULE)

    parser.add_argument('--noencode', dest='omitEncode',
	action='store_const', required=False, default=False, const=True,
    	help='omit Encode experiments w/ ID prefixes "%s" from the dataset'  \
						% ENCODE_ID_PREFIX)

    parser.add_argument('-o', '--outputDir', dest='outputDir', action='store',
	required=False, default=DATADIR,
    	help='parent dir where /no and /yes are created. Default=%s' % DATADIR)

    args = parser.parse_args()
    return args
  
# Main prog
def main():
    args = parseCmdLine()

    if args.preprocessor != None:
	preproc = getattr( textTuningLib, args.preprocessor )

    # for now assume directories exist
    # could create directories instead.

    fp = open( args.inputFile, 'r')
    for i, line in enumerate( fp.readlines()[1:] ):

	title, desc, ID, yesNo = map( string.strip, line.split('\t') )

	if args.omitEncode and title.find(ENCODE_ID_PREFIX) > -1:
	    print "Skipping ENCODE exp: '%s'" % ID
	    continue
	
	filename = os.sep.join( [ args.outputDir, yesNo.lower(), ID ] )

	with open(filename, 'w') as newFile:

	    # separate title and desc with punctuation just so we can
	    #  look in the files. sklearn vectorizer will ignore punctuation.
	    doc =  title + ' ----' + desc

	    if args.preprocessor != None:
		doc = preproc(doc)
	    newFile.write( doc )
#
main()
