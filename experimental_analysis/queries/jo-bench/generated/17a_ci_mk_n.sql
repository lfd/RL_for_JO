SELECT * FROM movie_keyword AS mk, name AS n, cast_info AS ci WHERE n.name LIKE 'B%' AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;