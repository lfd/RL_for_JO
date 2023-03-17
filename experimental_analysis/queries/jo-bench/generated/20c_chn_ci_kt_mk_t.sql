SELECT * FROM kind_type AS kt, title AS t, cast_info AS ci, char_name AS chn, movie_keyword AS mk WHERE chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND kt.kind = 'movie' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;