SELECT * FROM name AS n, aka_name AS an, cast_info AS ci, title AS t, movie_keyword AS mk, movie_companies AS mc WHERE an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;