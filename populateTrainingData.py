#!/usr/bin/env python2.7 
#
# populateTrainingData.py
# Script to take a sample file (tab-delimited) and split it
# data and split it into individual experiment files, one for each sample.
#
# Files (samples) get shoved into two directories, /data/no and /data/yes,
#       that sklearn.datasets.load_files() can read easily.
#
# Samples w/ eval state = No and Yes go into their respective directories.
#
# See sampleDataLib.py for input and output file formats.
# This script is intended to be independent of specific ML projects.
# The details of data samples are intended to be encapsulated in
#   sampleDataLib.py
#
# Author: Jim Kadin
#
import sys
sys.path.append('..')
sys.path.append('../..')
import string
import os
import argparse
from ConfigParser import ConfigParser
import sampleDataLib as sdLib
import sklearnHelperLib as ppLib	# module holding preprocessor function

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

TRAINING_DATA = cp.get("DEFAULT", "TRAINING_DATA")
PREPROCESSOR  = cp.get("DEFAULT", "PREPROCESSOR")

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Splits GXD HT training data into sklearn directory structure')

    parser.add_argument('inputFile', action='store', 
    	help='tab-delimited input file of training data')

    parser.add_argument('-o', '--outputDir', dest='outputDir', action='store',
	required=False, default=TRAINING_DATA,
    	help='parent dir where /no and /yes exist. Default=%s' % TRAINING_DATA)

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

    for yesNo in ['yes', 'no']:
	dirname =  os.sep.join( [ args.outputDir, yesNo ] )
	if not os.path.exists(dirname):
	    os.makedirs(dirname)

    counts = { 'yes':0, 'no':0, 'skipped':0}
    fp = open( args.inputFile, 'r')
    for line in fp.readlines()[1:]:

	sample = sdLib.SampleRecord(line, preprocessor=args.preprocessor,)
	rejectReason = sample.isReject()
	if rejectReason != None: 
	    print "skipping sample: %s" % rejectReason
	    counts['skipped'] += 1
	    continue
	
	yesNo = sample.getKnownClassName()
	counts[yesNo] += 1

	filename = os.sep.join([args.outputDir, yesNo, sample.getSampleName()])

	with open(filename, 'w') as newFile:
	    newFile.write(sample.getDocument()) 

    numFiles = counts['yes'] + counts['no']
    print "%d files written to %s" % (numFiles, args.outputDir)
    print "%d yes, %d no" % (counts['yes'], counts['no'])
    print "%d samples skipped" % counts['skipped']
#
main()
