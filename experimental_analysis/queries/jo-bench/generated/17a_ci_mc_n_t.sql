SELECT * FROM movie_companies AS mc, title AS t, cast_info AS ci, name AS n WHERE n.name LIKE 'B%' AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;