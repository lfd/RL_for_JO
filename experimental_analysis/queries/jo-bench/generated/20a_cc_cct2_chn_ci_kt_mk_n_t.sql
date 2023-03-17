SELECT * FROM movie_keyword AS mk, cast_info AS ci, name AS n, char_name AS chn, comp_cast_type AS cct2, complete_cast AS cc, title AS t, kind_type AS kt WHERE cct2.kind LIKE '%complete%' AND chn.name NOT LIKE '%Sherlock%' AND (chn.name LIKE '%Tony%Stark%' OR chn.name LIKE '%Iron%Man%') AND kt.kind = 'movie' AND t.production_year > 1950 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;