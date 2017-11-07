# Graphing confidence distributions for predictions

graphConfTrainOrTest = function(filename)
{
    # Graph conf distribution for predictions for the training and test sets.
    # Assume file is tab delimited, with header line,
    # Structure:
    #  Sample (name),
    #  True classification (0/1),
    #  Predicted classification (0/1),
    #  FN/FP (string: "FP", "FN", or ""),
    #  Confidence (float),
    #  Abs Value of the confidence (float)

    ds = read.table(filename, header=TRUE, sep='\t', row.names="Sample")

    truePreds = subset(ds, ds[,3] == "")	# true
    FP = subset(ds, ds[,3] == "FP")		# false positives
    FN = subset(ds, ds[,3] == "FN")		# false negatives
    FPFN = rbind(FP, FN)			# all FP FN together

    maxX = max(ds$Abs.value)
    ticksX = seq(floor(-maxX),ceiling(maxX),0.05)
    lwd0 = 1.5				# y-axis line width at x=0
    Ymult = 1.3
    title = paste("Confidence Values for Known Samples",filename,date(),sep="\n")

    par(mfrow = c(3,1))				# 3 graphs, 1 column
    maxY = max(hist(ds$Confidence,breaks=40,plot=FALSE)$counts)
    hist(ds$Confidence,breaks=40, xlim=c(-maxX,maxX), main=title, 
	    xlab="All Predictions", xaxt='n',ylim=c(0,Ymult*maxY), col='yellow')
    axis(1, at=ticksX)
    axis(2, pos=0, col='black', tck=0, lwd=lwd0, labels=FALSE) # y axis at 0

    maxY = max(hist(truePreds$Confidence,breaks=40,plot=FALSE)$counts)
    hist(truePreds$Confidence,breaks=40, xlim=c(-maxX,maxX), main="",
	    ylim=c(0,Ymult*maxY),xlab="True Predictions", xaxt='n', col='green')
    axis(1, at=ticksX)
    axis(2, pos=0, col='black', tck=0, lwd=lwd0, labels=FALSE) # y axis at 0
    
    maxY = max(hist(FPFN$Confidence,breaks=40,plot=FALSE)$counts)
    hist(FPFN$Confidence, breaks=40, xlim=c(-maxX,maxX), main="",
	    ylim=c(0,Ymult*maxY), xlab="FP and FN", xaxt='n', col='red')
    axis(1, at=ticksX)
    axis(2, pos=0, col='black', tck=0, lwd=lwd0, labels=FALSE) # y axis at 0
}

graphConfPredictions = function(filename)
{
    # Graph confidence distribution for predictions for unknown set
    #    (no "True" classifications are known)
    # Assume file is tab delimited, with header line,
    # Structure:
    #  Sample (name),
    #  Predicted classification (0/1),
    #  Confidence (float),
    #  Abs Value of the confidence (float)

    ds = read.table(filename, header=TRUE, sep='\t', row.names="Sample")

    lwd0 = 1.5				# y-axis line width at x=0
    Ymult = 1.3
    maxY = max(hist(ds$Confidence,breaks=40,plot=FALSE)$counts)
    maxX = max(ds$Abs.value) + .5
    title = paste("Confidence Values for Unknown Samples",filename,date(),sep="\n")

    # cannot for the life of me figure out how to display axis x=0, 
    #  there must be some way..

    # maxY=c(0,40),	# set maxY?
    hist(ds$Confidence,breaks=40, xlim=c(-maxX, maxX), main=title, 
	xlab="All Predictions", ylim=c(0,Ymult*maxY),col='blue')
    axis(2, pos=0, col='black', tck=0, lwd=lwd0, labels=FALSE) # y axis at 0
}
