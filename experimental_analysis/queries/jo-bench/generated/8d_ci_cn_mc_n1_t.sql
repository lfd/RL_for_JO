SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, cast_info AS ci, name AS n1 WHERE cn.country_code = '[us]' AND n1.id = ci.person_id AND ci.person_id = n1.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;