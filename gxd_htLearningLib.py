#!/usr/bin/env python2.7 
#
# Library to support handling of GXD HT experiment text documents
#  (title, desc, experimental factors)
# Author: Jim Kadin
#

import sys
sys.path.append('..')
sys.path.append('../..')
import string
import re
from ConfigParser import ConfigParser

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg"])

ENCODE_ID_PREFIX = cp.get("DEFAULT", "ENCODE_ID_PREFIX")
#----------------------

def isEncodeExperiment(title
    ):
    '''
    Return True if this is the title of an Encode experiment from ArrayExpress
    '''
    return title.find(ENCODE_ID_PREFIX) > -1
#----------------------

punct = re.compile( '[^a-z0-9_]' )      # anything not a letter, digit, _1
def convertExpFactorStr(factorStr,	# factor terms separated by '|'
    ):
    '''
    Split out '|' delimited experimental factor terms.
    Clean up each term:
	Smoosh exp factor terms so each will tokenize as a single token.
	(remove all spaces and convert all punct to "_")
	Add "_EF" to differentiate these terms from other words,
	(will also prevent stemming of the term, not sure that is good or bad)
    Return a string with all the cleaned up terms
    '''
    if factorStr == "None": return '--'
    terms = []
    for t in factorStr.split('|'):
	term = t.strip().lower()
	term = re.sub(punct, '_', term) + "_EF"
	terms.append(term)

    return ' '.join(terms)
#----------------------
  
def constructDoc(title, desc, factorStr):
    # '---' so it is easy to look at and see the boundaries
    return title + ' --- ' + desc + ' --- ' + convertExpFactorStr(factorStr)

