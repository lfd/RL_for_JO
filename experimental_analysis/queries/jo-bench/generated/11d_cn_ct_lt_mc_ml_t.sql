SELECT * FROM link_type AS lt, movie_link AS ml, title AS t, movie_companies AS mc, company_name AS cn, company_type AS ct WHERE cn.country_code != '[pl]' AND ct.kind != 'production companies' AND ct.kind IS NOT NULL AND mc.note IS NOT NULL AND t.production_year > 1950 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id;