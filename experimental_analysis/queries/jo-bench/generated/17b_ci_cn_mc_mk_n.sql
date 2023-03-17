SELECT * FROM movie_companies AS mc, movie_keyword AS mk, cast_info AS ci, name AS n, company_name AS cn WHERE n.name LIKE 'Z%' AND n.id = ci.person_id AND ci.person_id = n.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;