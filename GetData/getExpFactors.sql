-- get HT experiments and their experimental factor terms
select a.accid,  p.value
from mgi_property p, voc_term t, acc_accession a
where p._PropertyTerm_Key = t._Term_key
and t.term = 'raw experimental factor'
and p._MGITYpe_key = 42
and a._Object_key = p._Object_key
and a._mgitype_key = 42
order by a.accid, p.value
