SELECT ht._evaluatedby_key, u.name, t.term, count(*) AS "count"
FROM gxd_htexperiment ht join MGI_user u on (ht._evaluatedby_key = u._user_key)
    join voc_term t on (ht._evaluationstate_key = t._term_key)
GROUP BY ht._evaluatedby_key, u.name, t.term
order by u.name
