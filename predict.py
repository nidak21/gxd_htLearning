#!/usr/bin/env python2.7 
#
# Script to take a samples and predict them using a trained model
#
# Write out prediction file and optional a long prediction file.
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
import pickle
import argparse
from ConfigParser import ConfigParser
import sampleDataLib as sdLib
import sklearnHelperLib as ppLib	# module holding preprocessor function

#-----------------------------------
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg","../../config.cfg","../../../config.cfg"])

DATA_TO_PREDICT	 = cp.get("DEFAULT", "DATA_TO_PREDICT")
BLESSED_MODEL	 = cp.get("DEFAULT", "BLESSED_MODEL")
PREPROCESSOR     = cp.get("DEFAULT", "PREPROCESSOR")
DEFAULT_OUTPUT   = "predictions.tsv"
DEFAULT_OUTPUT_LONG  = "predictions_long.tsv"

def parseCmdLine():
    parser = argparse.ArgumentParser( \
		description='predict samples/records relevance')

    parser.add_argument('-i', '--input', dest='inputFile', action='store', 
	required=False, default=DATA_TO_PREDICT,
    	help='tab-delimited record input file. Default: %s' \
				% DATA_TO_PREDICT)

    parser.add_argument('-m', '--model', dest='model', action='store',
	required=False, default=BLESSED_MODEL,
    	help='pickled model file. Default: %s' % BLESSED_MODEL)

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

    parser.add_argument('-p', '--preprocessor', dest='preprocessor',
        action='store', required=False, default=PREPROCESSOR,
        help='preprocessor function name. Default= %s' % PREPROCESSOR)

    args = parser.parse_args()

    return args
#----------------------
  
def main():
    args = parseCmdLine()

    #####################
    # Get Trained Model
    with open(args.model, 'rb') as bp:
	model = pickle.load(bp)

    hasConf = hasattr(model, "decision_function")# confidence values?

    #####################
    # Read file of samples to predict
    print "Reading documents from %s...." % args.inputFile

    samples = []			# sample records
    docs = []				# documents (from samples)
    counts = { 'skipped':0, }
    
    ip = open(args.inputFile, 'r')

    for line in ip.readlines()[1:]:
	sample = sdLib.SampleRecord(line, preprocessor=args.preprocessor,
						    hasConfidence=hasConf)
	rejectReason = sample.isReject()
	if rejectReason != None:
	    print "skipping sample: %s" % rejectReason
	    counts['skipped'] += 1
            continue

	doc = sample.getDocument()
	docs.append(doc)   		# str(doc).strip()) ?
	samples.append(sample)
    print "...done %d documents. Skipped %d" % (len(docs), counts['skipped'])

    #####################
    # PREDICT!!
    print "Predicting...."
    y_predicted = model.predict(docs)
    if hasConf:
        confs = model.decision_function(docs).tolist()
    print "...done"

    #####################
    # Write Output Files
    print "Writing prediction file(s)..."

    fp = open(args.outputFile, 'w')
    fp.write(samples[0].getPredOutputHeader())

    if args.writeLong:
	lfp = open(args.longFile, 'w')
	lfp.write(samples[0].getPredLongOutputHeader())

    for i, (s, y) in enumerate(zip(samples, y_predicted, )):
	conf = None
	if hasConf: conf = confs[i]
	fp.write(s.getPredOutput(y, conf))
	if args.writeLong: lfp.write(s.getPredLongOutput(y, conf))

    print "...done %d lines written to %s" % (len(samples), args.outputFile)
    if args.writeLong:
	print "...done %d lines written to %s" % (len(samples), args.longFile)

    return
# ---------------------------
main()
