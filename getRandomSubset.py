#!/usr/bin/env python2.7 
#
# Produce a random subset of lines from a file (typically TSV file)
# Read from stdin, write random subset file and "leftover" file 
#  (the ones not selected at the random subset)
#  of lines.
# Author: Jim Kadin
#
import sys
import string
import random
import argparse
#-----------------------------------

def parseCmdLine():
    parser = argparse.ArgumentParser( \
	    description='Split lines from stdin into a random subset and leftovers. Output lines are in the same order as input.')

    parser.add_argument('-r', '--randomfile', dest='randomFile', action='store',
	required=True, help='output file for random lines')

    parser.add_argument('-l', '--leftoverfile', dest='leftoverFile',
	action='store', required=True, help='output file for leftover lines')

    parser.add_argument('--noheader', dest='hasHeader',
        action='store_false', required=False, 
        help='input has no header line to keep. Default: preserve header in output files')

    parser.add_argument('-n', '--num', dest='numKeep', action='store',
	required=False, type=int, default=0, help='number of rows to output')

    parser.add_argument('-f', '--fraction', dest='fractionKeep', action='store',
	required=False, type=float, default=0.0,
    	help='fraction of rows to output. Float between 0 and 1.')

    args = parser.parse_args()

    if args.numKeep == 0 and args.fractionKeep == 0.0:
	sys.stderr.write('error: you need to specify -n or -f\n\n')
	parser.print_usage(sys.stderr)
	exit(4)

    return args
#----------------------

# Main prog
def main():
    args = parseCmdLine()

    lines = sys.stdin.readlines()

    rfp = open(args.randomFile, 'w')
    lfp = open(args.leftoverFile, 'w')
    
    if args.hasHeader:
	rfp.write(lines[0])
	lfp.write(lines[0])
	del lines[0]

    if args.numKeep != 0:
	num = args.numKeep
    else:
	num = int(len(lines) * args.fractionKeep)

    # list of random line indexes, sorted
    randomLines = sorted(random.sample(range(len(lines)), num))

    numRlines = 0
    numLlines = 0

    for i in range(len(lines)):
	if len(randomLines) != 0 and i == randomLines[0]:
	    rfp.write(lines[i])
	    del randomLines[0]
	    numRlines += 1
	else:
	    lfp.write(lines[i])
	    numLlines += 1

    print "%d random lines written to %s" % (numRlines, args.randomFile)
    print "%d leftover lines written to %s" % (numLlines, args.leftoverFile)
    return

# ---------------------------
main()
