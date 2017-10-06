-- 1) get experimental factors for each experiment joined together with ' | '
create temporary table tmp_expFactors
as
select a.accid, string_agg(p.value, ' | ') factors
from mgi_property p, voc_term t, acc_accession a
where p._PropertyTerm_key = t._Term_key
and t.term = 'raw experimental factor'
and p._MGIType_key = 42
and a._Object_key = p._Object_key
and a._mgitype_key = 42
and a._logicaldb_key = 189
group by a.accid;

create index idx1 on tmp_expFactors(accid);

-- Get all HT experiments whose evaluation status is 'no' or 'yes'
select  a.accid "exp ID", t.term "eval state", ht.name "title", ht.description,
        f.factors
from gxd_htexperiment ht join acc_accession a
        on (a._object_key = ht._experiment_key
                            and a._logicaldb_key = 189) -- ArrayExpress log db
    join voc_term t on ( ht._evaluationstate_key = t._term_key)
    left join tmp_expFactors f on (a.accid = f.accid)
where t.term in ('Yes', 'No')
and ht._evaluatedby_key != 1561  -- not loader evaluation
                                 -- loader sets super series to "No"
