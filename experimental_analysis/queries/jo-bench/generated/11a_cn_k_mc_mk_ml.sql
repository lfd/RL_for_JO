SELECT * FROM keyword AS k, movie_link AS ml, movie_keyword AS mk, movie_companies AS mc, company_name AS cn WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND k.keyword = 'sequel' AND mc.note IS NULL AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;