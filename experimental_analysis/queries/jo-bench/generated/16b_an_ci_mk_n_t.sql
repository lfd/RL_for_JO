SELECT * FROM name AS n, aka_name AS an, movie_keyword AS mk, cast_info AS ci, title AS t WHERE an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;