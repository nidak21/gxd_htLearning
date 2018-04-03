
# pull together Connie's evaluation of GXD HT experiment relevance from
#   older files (split into yes and no files)
#   with newer predictions file
import sys
import string

evalNoFile = sys.argv[1]	# (old) predicted "no" evaluated by Connie
evalYesFile = sys.argv[2]	# (old) predicted "yes" evaluated by Connie
newPredFile= sys.argv[3]		# (new) predictions file


evalNoLines = open( evalNoFile, 'r').readlines()
evalYesLines = open( evalYesFile, 'r').readlines()
newPredLines = open(newPredFile, 'r').readlines()

# for computing confusion matrix and precision/recall for old and new preds
oldCounts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN':0}
newCounts = {'TP': 0, 'FP': 0, 'TN': 0, 'FN':0}
def updateCounts( evaluated, oldP, newP):
    if evaluated == 'yes':
	if oldP == 'yes': oldCounts['TP'] += 1
	if oldP == 'no':  oldCounts['FN'] += 1
	if newP != 'none':
	    if newP == 'yes': newCounts['TP'] += 1
	    if newP == 'no':  newCounts['FN'] += 1
    elif evaluated == 'no':
	if oldP == 'yes': oldCounts['FP'] += 1
	if oldP == 'no':  oldCounts['TN'] += 1
	if newP != 'none':
	    if newP == 'yes': newCounts['FP'] += 1
	    if newP == 'no':  newCounts['TN'] += 1
# end updateCounts


newPreds = {}		# build dict of (new) predictions
for line in newPredLines[1:]:
    parts = line.split('\t')
    newPreds[parts[0]] = {
			'AE ID'      : parts[0],
			'New Pred' : ['no', 'yes'][int(parts[1])],
    			'Confidence' : parts[2],
			'Abs value'  : parts[3],
			'Exp factors': parts[4],
			'Title'      : parts[5],
			}


evals = {}		# build dict of (old) no predictions evaluated by C
for line in evalNoLines[1:]:
    parts = line.split('\t')
    evals[parts[0]] = {
			'AE ID'     	: parts[0],
			'Exp type'	: parts[2],
			'Eval'		: parts[3].lower(),
			'Old Pred'	: 'no',
			}

			# add (old) yes predictions evaluated by C
for line in evalYesLines[1:]:
    parts = line.split('\t')
    evals[parts[0]] = {
			'AE ID'     	: parts[0],
			'Exp type'	: parts[2],
			'Eval'		: parts[3].lower(),
			'Old Pred'	: 'yes',
			}

# go through evaluated (old) predictions and match them up with the new
print '\t'.join([
		'AE ID',
		'Eval',
		'Old Pred',
		'New Pred',
		'Confidence',
		'Abs value',
		'Exp type'
		'Exp factors',
		#'Title',
		])
for (id, ev) in evals.items():
    if newPreds.has_key(id):	# evaluated record has a new prediction
	pr = newPreds[id]
	updateCounts( ev['Eval'], ev['Old Pred'], pr['New Pred'])
	print '\t'.join([
			id,
			ev['Eval'],
			ev['Old Pred'],
			pr['New Pred'],
			pr['Confidence'],
			pr['Abs value'],
			ev['Exp type'],
			pr['Exp factors'],
			#pr['Title'],
			])
	del newPreds[id]
    else:			# evaluted record has no new prediction
	updateCounts( ev['Eval'], ev['Old Pred'], 'none')
	print '\t'.join([
			id,
			ev['Eval'],
			ev['Old Pred'],
			'none',
			'none',
			'none',
			ev['Exp type'],
			'',
			#'',
			])

# report any new predictions that don't have evaluated, old predictions
print
print
print
print "New predictions that have not been evaluated: %d" % len(newPreds)
for (id, pr) in newPreds.items():
	print '\t'.join([
			id,
			'none',
			'none',
			pr['New Pred'],
			pr['Confidence'],
			pr['Abs value'],
			'none',
			pr['Exp factors'],
			'',
			#'',
			])

# report confusion matrics and precision/recall

print
print
print 'Old predictions'
print '\t Predicted'
print '\t Yes\t No'
print 'tru yes\t %d\t %d' % ( oldCounts['TP'], oldCounts['FN'])
print 'tru no\t %d\t %d' % ( oldCounts['FP'], oldCounts['TN'])
print
print 'Precision: %f\t Recall: %f' % \
	(
	 float(oldCounts['TP'])/float(oldCounts['TP']+oldCounts['FP']),
	 float(oldCounts['TP'])/float(oldCounts['TP']+oldCounts['FN']),
	)

print
print 'New predictions'
print '\t Predicted'
print '\t Yes\t No'
print 'tru yes\t %d\t %d' % ( newCounts['TP'], newCounts['FN'])
print 'tru no\t %d\t %d' % ( newCounts['FP'], newCounts['TN'])
print
print 'Precision: %f\t Recall: %f' % \
	(
	 float(newCounts['TP'])/float(newCounts['TP']+newCounts['FP']),
	 float(newCounts['TP'])/float(newCounts['TP']+newCounts['FN']),
	)
