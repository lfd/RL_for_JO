SELECT * FROM role_type AS rt, cast_info AS ci, company_name AS cn, movie_companies AS mc, title AS t, aka_name AS a1 WHERE cn.country_code = '[us]' AND rt.role = 'writer' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND a1.person_id = ci.person_id AND ci.person_id = a1.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;