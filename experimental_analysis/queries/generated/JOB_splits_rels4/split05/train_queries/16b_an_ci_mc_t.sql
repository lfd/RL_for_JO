SELECT * FROM title AS t, cast_info AS ci, aka_name AS an, movie_companies AS mc WHERE ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;