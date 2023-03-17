SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, link_type AS lt, movie_link AS ml, company_type AS ct, movie_companies AS mc, company_name AS cn, title AS t WHERE cct2.kind LIKE 'complete%' AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;