SELECT * FROM movie_companies AS mc, company_name AS cn, movie_info AS mi, movie_keyword AS mk, keyword AS k, movie_link AS ml WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND k.keyword = 'sequel' AND mc.note IS NULL AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;