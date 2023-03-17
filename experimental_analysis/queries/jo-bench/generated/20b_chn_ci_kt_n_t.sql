SELECT * FROM kind_type AS kt, title AS t, cast_info AS ci, char_name AS chn, name AS n WHERE chn.name NOT LIKE '%Sherlock%' AND (chn.name LIKE '%Tony%Stark%' OR chn.name LIKE '%Iron%Man%') AND kt.kind = 'movie' AND n.name LIKE '%Downey%Robert%' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id;