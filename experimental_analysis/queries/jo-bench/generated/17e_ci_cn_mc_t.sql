SELECT * FROM movie_companies AS mc, title AS t, cast_info AS ci, company_name AS cn WHERE cn.country_code = '[us]' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;