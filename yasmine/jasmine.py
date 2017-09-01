#Contains 3  helper functions: print results(confusion matrix, f-score,classification report) print falsely classfied experiments, and create custom scoring methods for grid search
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.metrics import f1_score,make_scorer,fbeta_score

POS_LABEL=1
LABELS=[1,0]

#Prints  classification report, confusion matrix and f-beta score for training and test set
def results(testset_actual_classification, predicted_for_test_set, trainset_actual_classification, predicted_for_train_set,beta):
     #Printing Classification Report for TestSet
     print("Test Set Classification Report & Confusion Matrix: \n")
     print(metrics.classification_report(testset_actual_classification, predicted_for_test_set,target_names=['yes','no'],labels=LABELS ))

     print "F %d: %5.3f"%(beta ,fbeta_score( testset_actual_classification, predicted_for_test_set,beta,pos_label=POS_LABEL))
     print("\n")

     #Printing Confusion Matrix for Test Set
     cm=metrics.confusion_matrix( testset_actual_classification, predicted_for_test_set ,  labels=LABELS)
     print(cm)
     print("\n")

     #Printing Classification Report for Training Set
     print("Training Set Classification Report & Confusion Matrix: \n")
     print(metrics.classification_report( trainset_actual_classification, predicted_for_train_set,target_names=['yes','no'],labels=LABELS, ))
     print "F %d: %5.3f"%(beta ,fbeta_score( trainset_actual_classification, predicted_for_train_set,beta,pos_label=POS_LABEL))

     #Printing Confusion Matrix for Training Set
     cm=metrics.confusion_matrix(  trainset_actual_classification, predicted_for_train_set,   labels=LABELS)
     print(cm)



#Prints the experiment ID of  the first ten False Negatives and False Positives
def falselyReported(y_test,  y_predicted_test, file_names_test):
    falsePositives= []
    falseNegatives= []
    
    for testset_classification,predicted_for_testset,expID in zip(y_test, y_predicted_test,file_names_test ):

        if testset_classification != predicted_for_testset:
            if predicted_for_testset == 1:
	    	falsePositives.append(expID)

            else:
		falseNegatives.append(expID)

    print "total count: ", len(falsePositives) + len(falseNegatives) 
    print

    print "Number of False Positves: ", len(falsePositives)
    print falsePositives[:11]
    print
    print "Number of False Negatives: ", len(falseNegatives)
    print falseNegatives[:11]

# Returns f-MYBETA scorer function
def make_f_scorer(MYBETA):
    if MYBETA==0:
        return None
    else:
        return make_scorer(fbeta_score, beta=MYBETA, pos_label=POS_LABEL)







