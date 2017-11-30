
import sys

f2Ids = {}
with open(sys.argv[2]) as f2:
    for line in f2.readlines():
	title, factors, desc, ID, state = line.split('\t')
	f2Ids[ID] = True
with open(sys.argv[1]) as f1:
    for line in f1.readlines():
	title, factors, desc, ID, state = line.split('\t')
	if f2Ids.has_key(ID) == False:
	    sys.stdout.write(line)
