-- Get all HT experiments whose evaluation status is 'no' or 'yes'
select  a.accid "exp ID", t.term "eval state", ht.name "title", ht.description
from gxd_htexperiment ht join acc_accession a
	on (a._object_key = ht._experiment_key)
    join voc_term t on ( ht._evaluationstate_key = t._term_key)
where a._logicaldb_key = 189	-- ArrayExpress logical db

and t.term in ('Yes', 'No')

and ht._evaluatedby_key != 1561  -- not loader evaluation
                                 -- loader sets super series to "No"
