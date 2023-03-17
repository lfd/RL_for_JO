SELECT * FROM title AS t, cast_info AS ci, name AS n, movie_companies AS mc, company_name AS cn WHERE n.name LIKE 'Z%' AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;