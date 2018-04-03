April 2, 2018

Working to reconcile Connie's recently manually evaluted experiments with my
older predicted evaluations.

What is up:
    - I had generated an initial set of predictions in October based on
	training 1615 samples previously evaluated by Connie

    - I continued tuning/improving the model and retrained on later data in
      early Nov (Nov 2) - this contained 3201 samples - and I generated a
      new set of predictions.  11563 newer predictions in total

    - Unfortunately, Connie and I miscommunicated, and she evaluted many of
      the older predictions to see how they came out rather than working
      from the later set.

    - The predictions from the older set:
    	4898 predicted yes
	    Connie evaluated these,
	    1871 true yes
	    3027 true no - not so good, 38% precision
	7330 predicted no
	    Connie evaluated only 169 of these
	    167 true no
	      2 true yes   - pretty good, but a small set.

    - So the work in this directory is to pull together her evaluated, older
      predictions with the later predictions to see how they compare.
      Hopefully for the experiments she had manually evaluted, the newer
      predictions were better than the older.
Files:
    predNo.txt  = the set of old "no" predictions that she evaluated (169)
    predYes.txt = the set of old "yes" predictions that she evaluated (4898)
    predictions_long.txt = the set of newer predictions
			   (note some of the samples that she evaluated were
			   in the training set for the newer model, so these
			   do not have newer predictions)

    combine.py - script that merges these and computes precision/recall

    combined.txt - the output of combine.py.  This has columns for
		    "Evaluated", "Old Prediction", "New Prediction"
		    so these can be compared

Here is the output of the comparison:
Old predictions
         Predicted
         Yes     No
tru yes  1864    2
tru no   3027    167

Precision: 0.381108      Recall: 0.998928

New predictions
         Predicted
         Yes     No
tru yes  1489    186
tru no   1256    1447

Precision: 0.542441      Recall: 0.888955

So a pretty good improvement on precision with a little loss of recall.
I consider this to mean the newer model is better.
