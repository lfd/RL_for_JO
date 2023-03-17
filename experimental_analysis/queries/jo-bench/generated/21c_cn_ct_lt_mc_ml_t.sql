SELECT * FROM movie_link AS ml, title AS t, link_type AS lt, movie_companies AS mc, company_name AS cn, company_type AS ct WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id;