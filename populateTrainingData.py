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
import re
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

    parser.add_argument('--expFactors', dest='expFactor', action='store',
	required=False, default=None,
    	help='file mapping experiments to experimental-factors')

    args = parser.parse_args()
    return args
#----------------------
# Experimental factors:
# These are terms associated with experiments by ArrayExpress curators.
# NOT a controlled vocab - there are >1000 distinct terms.
#
# Want to see if these help distinguish relevant/irrelevant experiments.
# Can think of several ways to include these:
# 1) treat the terms as a separate text field from the title/description and
#    tokenize to get separate features for the tokens in these terms.
#    (this seemed quite involved to do, but seems useful IF these annotations
#     are really significant. I.e., if an exp factor term of "development"
#     carries more weight than "development" in the regular text.)
# 2) just throw these terms into the text/description text and tokenize these
#    altogether. So the distinction that  individual tokens are part of
#     exp Factors is lost.
# 3) Something in the middle: smoosh the tokens of these factor terms into
#    indivisible tokens and add some suffix to differentiate these from
#    regular text features. Then add these "terms" to the title/desc text.
# (3) is what we are trying here for now. (probably won't make any difference!)
#----------------------

def getExpFactors(fileName):
    '''
    Return dict mapping experiment IDs to a set of their exp factors (strings)
    '''
    expToFactor = {}	# expToFactor[ expID ] = set of converted factor terms

    with open(fileName, 'r') as fp:
	for line in fp.readlines()[1:]:
	    expId, term = line.split('\t', 1)
	    term = convertExpFactorTerm(term)
	    expToFactor.setdefault( expId, set()).add(term)
    return expToFactor
#----------------------
punct = re.compile( '[^a-z0-9_]' ) 	# anything not a letter, digit, _

def convertExpFactorTerm(term):
    '''
    Smoosh exp factor terms so each will tokenize as a single token.
    (remove all spaces and convert all punct to "_")
    Add "_EF" to differentiate these terms from other words,
    (will also prevent stemming of the term, not sure that is good or bad)
    '''
    term = term.strip().lower()
    term = re.sub(punct, '_', term) + "_EF"
    return term
#----------------------
  
# Main prog
def main():
    args = parseCmdLine()

    if args.preprocessor != None:
	preproc = getattr( textTuningLib, args.preprocessor )

    if args.expFactor != None:
	expToFactors = getExpFactors(args.expFactor)

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
	    doc =  title + ' ---- ' + desc

	    if args.expFactor != None:
		factorsText = ' '.join(expToFactors.setdefault(ID,set()))
		doc += ' ' + factorsText
		#if factorsText != '': print "%s '%s'" % (ID,doc)

	    if args.preprocessor != None:
		doc = preproc(doc)
	    newFile.write( doc )
#
main()
