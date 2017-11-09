#!/usr/bin/env python2.7 
# Compare some sklearn Pipelines to each other over multiple
#  train_test_splits().
# Computes Fscore, precision, recall over multiple splits and then
#  computes averages across the splits.
# Also tries "voting" across the different Pipelines to see if that fares
#  better
#
import sys
import argparse
from ConfigParser import ConfigParser
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

#-----------------------------------
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

TRAINING_DATA = cp.get("DEFAULT", "TRAINING_DATA")
BETA          = cp.getint("MODEL_TUNING", "COMPARE_BETA")
INDEX_OF_YES  = cp.getint("DEFAULT", "INDEX_OF_YES")
CLASS_NAMES   = eval( cp.get("DEFAULT", "CLASS_NAMES") )

NUMSPLITS     = '5'
TESTSIZE      = '20'
PIPELINE_DEFS = 'goodPipelines.py'
#----------------------

def parseCmdLine():
    parser = argparse.ArgumentParser(description = \
		    'Compare pipelines. Write output to stdout.')

    parser.add_argument('-d', '--data', dest='trainingData', action='store', 
        required=False, default=TRAINING_DATA,
        help='Directory where training data files live. Default: %s' \
						    % TRAINING_DATA)
    parser.add_argument('-p','--pipelines',dest='pipelineDefs', action='store',
        required=False, default=PIPELINE_DEFS,
        help='Python file that defines "pipelines" list. Default: %s' \
						    % PIPELINE_DEFS)
    parser.add_argument('-s','--splits',dest='numSplits', action='store',
        required=False, default=NUMSPLITS,
        help='number of train-test splits to run. Default: %s' % NUMSPLITS)

    parser.add_argument('-t','--testsize',dest='testSize', action='store',
        required=False, default=TESTSIZE,
        help='percent of samples to use for test set. Default: %s' % TESTSIZE)

    args = parser.parse_args()
    args.numSplits = int(args.numSplits)
    args.testSize = float(args.testSize)/100.0
    return args
#----------------------
  
def main():
    args = parseCmdLine()
    execfile(args.pipelineDefs)		# define pipelines, list of Pipelines

    # totals across all the split tries for each pipeline + voted predictions
    #  for computing averages
    pipelineInfo = [ {	'fscores':0,
			'precisions': 0,
			'recalls': 0, } for i in range(len(pipelines)+1) ]
						# +1 for voted predictions
    dataSet = load_files( args.trainingData )

    for sp in range(args.numSplits):
	docs_train, docs_test, y_train, y_test = \
		train_test_split( dataSet.data, dataSet.target,
				test_size=args.testSize, random_state=None)

	predictions = []	# predictions[i]= predictions for ith Pipeline
				#  on this split (for voting)
	print "Sample Split"
	for i, pl in enumerate(pipelines):	# for each Pipeline

	    pl.fit(docs_train, y_train)
	    y_pred = pl.predict(docs_test)
	    predictions.append(y_pred)

	    precision, recall, fscore, support = \
			    precision_recall_fscore_support( \
							y_test, y_pred, BETA,
							pos_label=INDEX_OF_YES,
							average='binary')
	    pipelineInfo[i]['fscores']    += fscore
	    pipelineInfo[i]['precisions'] += precision
	    pipelineInfo[i]['recalls']    += recall

	    l="Pipeline %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			    % (i, BETA, fscore, precision, recall)
	    print l

	vote_pred = y_vote( predictions )
	precision, recall, fscore, support = \
			    precision_recall_fscore_support( \
							y_test, vote_pred, BETA,
							pos_label=INDEX_OF_YES,
							average='binary')
	i = len(pipelines)
	pipelineInfo[i]['fscores']    += fscore
	pipelineInfo[i]['precisions'] += precision
	pipelineInfo[i]['recalls']    += recall

	l="Votes    %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			% (i , BETA, fscore, precision, recall)
	print l

    # averages across all the Splits
    print
    for i in range(len(pipelines)+1):
	avgFscore    = pipelineInfo[i]['fscores']    / args.numSplits
	avgPrecision = pipelineInfo[i]['precisions'] / args.numSplits
	avgRecall    = pipelineInfo[i]['recalls']    / args.numSplits
	l="Average  %d: F%d: %6.4f\t precision: %4.2f\t recall: %4.2f" \
			% (i, BETA, avgFscore, avgPrecision, avgRecall)
	print l
#-----------------------

def y_vote( theYs,	# [ [y1's], [y2's], ...] parallel arrays of class assn's
	    ):
    '''
    Assuming each yi is an list of 0 and 1's,
    Return a parallel list that is the "vote" across all the yi's
    Ties default to 0 at this point...
    '''
    # there must be a better way to do this... 
    numOnes = theYs[0]	# numOnes[i] will be the number of 1's across y's[i]

    for Y in theYs[1:]:
	for i, val in enumerate(Y):
	    numOnes[i] += val

    votes = [ 0 for i in range(len(numOnes)) ]
    threshold = len(theYs)/2

    for i, c in enumerate(numOnes):
	if numOnes[i] > threshold: votes[i] = 1

    return votes
#-----------------------
if __name__ == "__main__": main()
