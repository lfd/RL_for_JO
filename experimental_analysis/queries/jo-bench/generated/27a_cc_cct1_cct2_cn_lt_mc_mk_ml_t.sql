SELECT * FROM movie_keyword AS mk, comp_cast_type AS cct1, link_type AS lt, movie_link AS ml, complete_cast AS cc, movie_companies AS mc, comp_cast_type AS cct2, company_name AS cn, title AS t WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;