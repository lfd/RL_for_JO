SELECT * FROM complete_cast AS cc, cast_info AS ci, char_name AS chn, title AS t, kind_type AS kt, comp_cast_type AS cct2 WHERE cct2.kind LIKE '%complete%' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND kt.kind = 'movie' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;