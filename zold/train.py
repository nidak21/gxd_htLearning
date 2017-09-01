"""
Play with classifying GXD high throughput experiment indexing using
scikit-learn.

Inputs are text files with Array Express experiment titles and descriptions.
Try to classify them as 'yes' or 'no' (relevant to GXD or not)
"""
# Author: Jim Kadin

#import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gxd_htLearningLib import getTrainingSet, getFormatedMetrics


dataset = getTrainingSet()

docs_train, docs_test, y_train, y_test = train_test_split( dataset.data,
						dataset.target, test_size=0.5)

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens. 1, 2, and 3 word ngrams.
cv = TfidfVectorizer(analyzer='word',ngram_range=(1,3), stop_words='english' )
			# option: strip_accents='ascii' or 'unicode' or 'None'j

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clasfer = Perceptron()
clf = Pipeline( [ ('vect', cv), ('clf', clasfer) ] )

# TASK: Fit the pipeline on the training set
clf = clf.fit( docs_train, y_train )

# TASK: Predict the outcome on the testing set in a variable named y_predicted
y_predicted = clf.predict( docs_test )

# Print the classification report
print getFormatedMetrics(y_test, y_predicted, dataset.target_names)

# display confusion matrix as a plot
import matplotlib.pyplot as plt
cm = metrics.confusion_matrix(y_test, y_predicted)
if False: # simple view
    plt.matshow(cm, cmap=plt.cm.jet)
    plt.show()

if True: # more coller view (don't know how to configure it)
    fig, ax = plt.subplots( figsize=(3, 2.5) )
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
	for j in range(cm.shape[1]):
	    ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
	#ax.set_xlabel(dataset.target_names[i]) # set axis labels, doesn't work
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

# Predict the result on some new text:
inputs = [
	'''Transcription profiling of human, chimp and mouse brain
	Microarray technologies allow the identification of large numbers of
	expression differences within and between species. Although
	environmental and physiological stimuli are clearly responsible for
	changes in the expression levels of many genes, it is not known
	whether the majority of changes of gene expression fixed during
	evolution between species and between various tissues within a
	species are caused by Darwinian selection or by stochastic
	processes. We find the following: (1) expression differences between
	species accumulate approximately linearly with time; (2) gene
	expression variation among individuals within a species correlates
	positively with expression divergence between species; (3) rates of
	expression divergence between species do not differ significantly
	between intact genes and expressed pseudogenes; (4) expression
	differences between brain regions within a species have accumulated
	approximately linearly with time since these regions emerged during
	evolution. These results suggest that the majority of expression
	differences observed between species are selectively neutral or
	nearly neutral and likely to be of little or no functional
	significance. Therefore, the identification of gene expression
	differences between species fixed by selection should be based on
	null hypotheses assuming functional neutrality. Furthermore, it may
	be possible to apply a molecular clock based on expression
	differences to infer the evolutionary history of tissues.''',
	'''
	Transcription profiling by array of human B cells and mouse NIH-3T3
	cells
	This SuperSeries is composed of the following subset Series:;
	GSE9973: Half-life determination for human B-cells (BL41); GSE9975:
	newly transcribed RNA (nt-RNA) for IFN alpha and gamma time course;
	GSE9977: Expression data from NIH-3T3 cells treated with mock, 100
	U/ml IFN alpha or 100 U/ml gamma for 1or 3h; GSE10011: Expression
	data from NIH-3T3 cells used for half-life determination Experiment
	Overall Design: Refer to individual Series'''
	]

predicted = clf.predict(inputs)

for s, p in zip(inputs, predicted):
    print(u'%s for\n"%s"' % (dataset.target_names[p],s ))
