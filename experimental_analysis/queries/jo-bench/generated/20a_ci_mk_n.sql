SELECT * FROM cast_info AS ci, name AS n, movie_keyword AS mk WHERE mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;