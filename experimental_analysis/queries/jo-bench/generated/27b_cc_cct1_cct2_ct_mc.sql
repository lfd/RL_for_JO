SELECT * FROM company_type AS ct, movie_companies AS mc, complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2 WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND ct.kind = 'production companies' AND mc.note IS NULL AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;