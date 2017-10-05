SELECT   t.term, count(*) AS "count", u.name
FROM gxd_htexperiment ht join voc_term t on (ht._evaluationstate_key=t._term_key)
left join MGI_user u on (ht._evaluatedby_key = u._user_key)
GROUP BY  t.term, u.name
order by  u.name, t.term
