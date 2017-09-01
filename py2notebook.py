#!/usr/bin/env python2.7 
'''
Convert a python file to a jupyter notebook.
See: 
https://stackoverflow.com/questions/23292242/converting-to-not-from-ipython-notebook-format

This conversion is pretty dumb: the python file is included as one big
cell in the notebook instead of breaking it up into multiple cells based on
the cell boundary comments.

But at least it lets you pull the script into jupyter where you can run it,
save it as HTML (including any generated graphs), and share that HTML
with people.
'''

import os.path as path
import argparse
from nbformat import v3,v4

def parseCmdLine():
    parser = argparse.ArgumentParser( \
    description='Converts a python file to a jupyter notebook file.')

    parser.add_argument('fileNames', action='store', nargs='+',
        help='1st is py file. 2nd is output filename, default is pyfile.ipynb'
	)
    args = parser.parse_args()

    setattr(args, 'inputFile', args.fileNames[0])
    if len(args.fileNames) > 1:
	setattr(args, 'outputFile', args.fileNames[1])
    else:
	setattr(args, 'outputFile',
		path.splitext(args.inputFile)[0] + '.ipynb' )
    return args

def main():
    args = parseCmdLine()

    with open(args.inputFile) as fpin:
	text = fpin.read()

    nbook = v3.reads_py(text)
    nbook = v4.upgrade(nbook)  # Upgrade v3 to v4

    jsonform = v4.writes(nbook) + "\n"
    with open(args.outputFile, "w") as fpout:
	fpout.write(jsonform)

main()
