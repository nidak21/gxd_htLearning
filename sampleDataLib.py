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
import sklearnHelperLib as ppLib	# holds doc preprocessor functions

#-----------------------------------
# Load config
cp = ConfigParser()
cp.optionxform = str # make keys case sensitive
cp.read(["config.cfg","../config.cfg","../../config.cfg","../../../config.cfg"])

PREPROCESSOR  = cp.get("DEFAULT", "PREPROCESSOR")
ENCODE_ID_PREFIX = cp.get("DEFAULT", "ENCODE_ID_PREFIX")
KEEP_ENCODE   = cp.getboolean("DEFAULT", "KEEP_ENCODE")

# As is the sklearn convention we use
#  y_true to be the index of the known class of a sample (from training set)
#  y_pred is the index of the predicted class of a sample/record
# CLASS_NAMES maps indexes to class names. CLASS_NAME[0] is 0th class name,etc.
CLASS_NAMES      = eval( cp.get("DEFAULT", "CLASS_NAMES") )

class SampleRecord (object):
    """
    Takes an inputLine (string) and parses it into a sample w/ attributes, 
	and provides access to those attibutes.
    So knows the format of a sample record from a TSV file.
    Knows how to format a record output line for various types of prediction
	reports.
    """
    # work out class name vs 0/1
    # what about Args and Config:
    #  access Config
    #  whatever other args should be passed in:  hasConfidence, preprocessor
    #   skip encode

    def __init__(self, line,
	preprocessor=PREPROCESSOR, # doc preprocessor funct name (or 'None')
	hasConfidence=False,
	):

	self.hasConfidence = hasConfidence	# does classifier have ability
						# to give prediction confidence
						# (affects output formats)

	fields = map(string.strip, line.split('\t'))
	if len(fields) == 5:	# have true Y value in input
	    self.ID           = fields[3]
	    self.className    = fields[4].lower()
	    self.expFactorStr = fields[1]
	    self.title        = fields[0]
	    self.desc         = fields[2]
	else:				# no true Y
	    self.ID           = fields[2]
	    self.className    = None
	    self.expFactorStr = fields[0]
	    self.title        = fields[3]
	    self.desc         = fields[1]

	self.doc = constructDoc(self.title, self.desc, self.expFactorStr)
	if preprocessor == 'None' or preprocessor == None:
	    self.preprocessor = None
	else:
	    self.preprocessor = getattr( ppLib, preprocessor )
	    self.doc = string.strip(self.preprocessor(self.doc))


    def getSampleName(self):
	return self.ID

    def getDocument(self):
	return self.doc

    def isReject(self):
	# Should this sample be kept or skipped????
	# Return None (do not reject) or a reason (string) why rejected
	if not KEEP_ENCODE and isEncodeExperiment(self.title):
	    return "Encode experiment %s" % self.ID
	else: return None

    def hasKnownClassName(self):
	return self.className != None

    def getKnownClassName(self):
	return self.className

    def getPredOutputHeader(self):
	cols = [ "Experiment" ]
	if self.className != None: cols.append("True Class")
	cols.append("Pred Class")
	if self.hasConfidence:
	    cols.append("Confidence")
	    cols.append("Abs Value")
	return '\t'.join(cols) + '\n'

    def getPredOutput(self, y_pred, confidence=None):
	# y_pred is the predicted class index, not the class name
	cols = [ self.ID ]
	if self.className != None:  cols.append(self.className)
	cols.append(CLASS_NAMES[y_pred])
	if self.hasConfidence:
	    cols.append("%6.3f" % confidence)
	    cols.append("%6.3f" % abs(confidence))
	return '\t'.join(cols) + '\n'

    def getPredLongOutputHeader(self):
	cols = [ "Experiment" ]
	if self.className != None: cols.append("True Class")
	cols.append("Pred Class")
	if self.hasConfidence:
	    cols.append("Confidence")
	    cols.append("Abs Value")
	cols.append("Exp Factors")
	cols.append("Title")
	cols.append("Description")
	cols.append("Processed Doc")
	return '\t'.join(cols) + '\n'

    def getPredLongOutput(self, y_pred, confidence=None):
	# include confidence value if not None
	cols = [ self.ID ]
	if self.className != None:  cols.append(self.className)
	cols.append(CLASS_NAMES[y_pred])
	if self.hasConfidence:
	    cols.append("%6.3f" % confidence)
	    cols.append("%6.3f" % abs(confidence))
	cols.append(self.expFactorStr)
	cols.append(self.title)
	cols.append(str(self.desc))
	cols.append(str(self.doc))
	return '\t'.join(cols) + '\n'

# end class SampleRecord ------------------------

#----------------------
# Experimental factors:
# These are terms associated with experiments by ArrayExpress curators.
# NOT a controlled vocab - there are >1000 distinct terms.
#
# These help distinguish relevant/irrelevant experiments.
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
