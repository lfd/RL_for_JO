SELECT * FROM link_type AS lt, movie_link AS ml, movie_companies AS mc, company_name AS cn, movie_keyword AS mk WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND lt.link LIKE '%follow%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;