SELECT * FROM name AS n, cast_info AS ci, aka_name AS an, movie_keyword AS mk WHERE an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;