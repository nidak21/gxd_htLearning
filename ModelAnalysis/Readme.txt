April 2, 2018
Since Connie has evaluted more ArrayExpress experiments, we have more training
data. So I experimented with retraining.

----------------------------------------------------
Without balancing "no" and "yes"
7625 evaluated experiments
2794 yes
4831 no

LinearSVC
best I could get was around  precision 74    recall 86

----------------------------------------------------
With balancing
Pick random set of 2800 "no"s and training with that and 2794 "yes"s

LinearSVC
Get aroung precision 79%   recall 89%

Running predictions on the  2031 leftover "no"s, get 1410 predicted to be "no"
so 70% of them predicted correctly

Seems reasonably good

Looking at confidence values for the leftover "no"s:
    the FP are mostly < 0.1, but so are the TN's so no apparent difference
