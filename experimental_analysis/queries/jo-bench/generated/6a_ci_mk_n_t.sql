SELECT * FROM name AS n, cast_info AS ci, title AS t, movie_keyword AS mk WHERE n.name LIKE '%Downey%Robert%' AND t.production_year > 2010 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;