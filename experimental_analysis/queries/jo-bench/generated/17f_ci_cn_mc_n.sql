SELECT * FROM name AS n, company_name AS cn, movie_companies AS mc, cast_info AS ci WHERE n.name LIKE '%B%' AND n.id = ci.person_id AND ci.person_id = n.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;