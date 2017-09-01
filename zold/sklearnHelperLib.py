'''
Some helper functions for using sklearn
'''
def orderTargets( bunch,	# dict w/ keys: target_names (list of names)
				#    target (list of indexes into target_names)
		  targetOrder	# list, reordering of the target_names
    ):
    ''' reorder the target labels in a sklearn bunch so they do not
        depend on the label alpha order.
    '''
    targetMapping = []
    for t in bunch.target_names:
	targetMapping.append( targetOrder.index(t) )

    for n in range(len(bunch.target)):
	bunch.target[n] = targetMapping[ bunch.target[n] ]
	# might be some way to do this more efficiently with numpy,
	#  need to learn about that.

    bunch.target_names = targetOrder
# end orderTargets

