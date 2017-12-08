-- Get all unevaluated HT experiments to predict relevance for

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

-- 2) pull title, description, experimental factors together.
select  a.accid "exp ID", f.factors, ht.name "title",
		translate(ht.description, E'\t\n', '  ') "description"
from gxd_htexperiment ht
    join acc_accession a  on (a._object_key = ht._experiment_key
				and a._logicaldb_key = 189) -- ArrayExpress
    left join tmp_expFactors f on (a.accid = f.accid)
where a._logicaldb_key = 189	-- ArrayExpress logical db
and ht._evaluationstate_key = 20225941 -- "Not Evaluated"
order by a.accid
