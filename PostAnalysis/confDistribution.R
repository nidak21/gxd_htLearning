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

    par(mfrow = c(3,1))				# 3 graphs, 1 column
    maxX = max(ds$Abs.value) + .5
    title = paste("Confidence Values",filename,date(),sep="\n")

    # cannot for the life of me figure out how to display axis x=0, 
    #  there must be some way..

    # maxY=c(0,40),	# set maxY?
    hist(ds$Confidence,breaks=40, xlim=c(-maxX,maxX), main=title, 
    			xlab="All Predictions")
    hist(truePreds$Confidence,breaks=40, xlim=c(-maxX,maxX), main="", 
    			xlab="True Predictions")
    hist(FPFN$Confidence, breaks=40, xlim=c(-maxX,maxX), main="",
			xlab="FP and FN")
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

    maxX = max(ds$Abs.value) + .5
    title = paste("Confidence Values",filename,date(),sep="\n")

    # cannot for the life of me figure out how to display axis x=0, 
    #  there must be some way..

    # maxY=c(0,40),	# set maxY?
    hist(ds$Confidence,breaks=40, xlim=c(-maxX, maxX), main=title, 
    			xlab="All Predictions")
}
