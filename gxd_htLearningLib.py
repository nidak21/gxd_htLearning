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
# Experimental factors:
# These are terms associated with experiments by ArrayExpress curators.
# NOT a controlled vocab - there are >1000 distinct terms.
#
# Want to see if these help distinguish relevant/irrelevant experiments.
# Can think of several ways to include these:
# 1) treat the terms as a separate text field from the title/description and
#    tokenize to get separate features for the tokens in these terms.
#    (this seemed quite involved to do, but seems useful IF these annotations
#     are really significant. I.e., if an exp factor term of "development"
#     carries more weight than "development" in the regular text.)
# 2) just throw these terms into the text/description text and tokenize these
#    altogether. So the distinction that  individual tokens are part of
#     exp Factors is lost.
# 3) Something in the middle: smoosh the tokens of these factor terms into
#    indivisible tokens and add some suffix to differentiate these from
#    regular text features. Then add these "terms" to the title/desc text.
# (3) is what we are trying here for now.
#----------------------

punct = re.compile( '[^a-z0-9_]' )      # anything not a letter, digit, _1
def convertExpFactorStr(factorStr,	# factor terms separated by '|'
    ):
    '''
    Split out '|' delimited experimental factor terms.
    Clean up each term:
	Smoosh exp factor terms so each will tokenize as a single token.
	I.e., remove all spaces and convert all punct to "_".
	Add "_EF" to differentiate these terms from other words,
    Return a string with all the cleaned up terms
    '''
    if factorStr == "None" or factorStr == "" or factorStr == None: return ""

    terms = []
    for t in factorStr.split('|'):
	term = t.strip().lower()
	term = re.sub(punct, '_', term) + "_EF"
	# squeeze all sequences of '_' down to one
	term = '_'.join( [ t for t in term.split('_') if len(t)>1 ] )
	terms.append(term)

    return ' '.join(sorted(terms))
#----------------------
  
def constructDoc(title, desc, factorStr):
    # '---' so it is easy to look at and see the boundaries
    return title + ' --- ' + desc + ' --- ' + convertExpFactorStr(factorStr)
#----------------------

def isEncodeExperiment(title
    ):
    '''
    Return True if this is the title of an Encode experiment from ArrayExpress
    '''
    return title.find(ENCODE_ID_PREFIX) > -1
#----------------------
