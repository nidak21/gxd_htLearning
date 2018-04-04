# Graphing confidence distributions for predictions

graphConfTrainOrTest = function(filename)
{
    # Graph confidence distributions for predictions for training or test sets.
    # Assume file is tab delimited, with header line,
    # Structure:
    #  Sample (string),
    #  True classification (not used),
    #  Predicted classification (not used),
    #  FN/FP (string: "FP", "FN", or for positives: "", "TP", or "TN"),
    #  Confidence (float),
    #  Abs value of the confidence (float)
    #
    # column names: Sample, FN/FP, Confidence, Abs value must have these
    #   exact names
    #
    # Display 3 distribution histograms in one output window:
    #   distribution of confidences for all predictions
    #   distribution of confidences for true predictions
    #   distribution of confidences for false (FN or FP) predictions
    # All three have the same x axis for comparison (but y axes differ)

    ds = read.table(filename, header=TRUE, sep='\t', row.names="Sample")

    # define relevant subsets of the data
    # True predictions
    truePreds = subset(ds, ds$FP.FN == "" | ds$FP.FN == "TP" | ds$FP.FN == "TN")
    # False predictions
    FP = subset(ds, ds$FP.FN == "FP")		# false positives
    FN = subset(ds, ds$FP.FN == "FN")		# false negatives
    FPFN = rbind(FP, FN)			# all FP FN together

    # define x axis params
    xMax = max(ds$Abs.value)
    xTicks = seq(floor(-xMax),ceiling(xMax),0.05) # where tick lines go
    histGroups = 40		# number of histogram boxes

    # define y axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # set up graphs (3 per page)
    title =paste("Prediction Confidence Distributions",filename,date(),sep="\n")
    par(mfrow = c(3,1))				# 3 graphs, 1 column

    # for each graph, we run hist to get the histogram values (without plotting)
    #  so we can get the yMax for the graph.
    # We use yMax to customize the y axis.
    # Then we run hist again to actually to the plot.

    # plot all predictions
    yMax = max(hist(ds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(ds$Confidence,breaks=histGroups, xlim=c(-xMax,xMax), main=title, 
	    xlab="All Predictions", xaxt='n',ylim=c(0,yMult*yMax), col='yellow')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0

    # plot true predictions
    yMax = max(hist(truePreds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(truePreds$Confidence,breaks=histGroups, xlim=c(-xMax,xMax), main="",
	    ylim=c(0,yMult*yMax),
		    xlab="True Negative & True Positive Predictions",
		    xaxt='n', col='green')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
    
    # plot false predictions
    yMax = max(hist(FPFN$Confidence,breaks=histGroups,plot=FALSE)$counts)

    hist(FPFN$Confidence, breaks=histGroups, xlim=c(-xMax,xMax), main="",
	    ylim=c(0,yMult*yMax),
		    xlab="False Negative and False Positive Predictions",
		    xaxt='n', col='red')
    axis(1, at=xTicks)
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}

graphConfPredictions = function(filename, xmax=0)
{
    # Graph confidence distribution for predictions for unknown set
    #    (no "True" classifications are known)
    # Assume file is tab delimited, with header line,
    # Structure:
    #  Sample (string),
    #  Predicted classification (0/1),
    #  Confidence (float),
    #  Abs value of the confidence (float)

    ds = read.table(filename, header=TRUE, sep='\t', row.names="Sample")

    # x-axis params
    if (xmax==0)xMax = max(ds$Abs.value) + .5
    else	xMax = xmax
    histGroups = 40		# number of histogram boxes

    # y-axis params
    yLwd0 = 1.5		# y-axis line width at x=0
    yMult = 1.3		# y max val multiplier to get y axis to stick up a bit

    # find yMax from the histogram (w/o plotting)
    yMax = max(hist(ds$Confidence,breaks=histGroups,plot=FALSE)$counts)

    title = paste("Prediction Confidence Distribution",
		    filename,date(),sep="\n")
    hist(ds$Confidence,breaks=histGroups, xlim=c(-xMax, xMax), main=title, 
	xlab="Confidence", ylim=c(0,yMult*yMax),col='blue')
    axis(2, pos=0, col='black', tck=0, lwd=yLwd0, labels=FALSE) # y axis at 0
}
