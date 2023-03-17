SELECT * FROM cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct2, char_name AS chn, movie_keyword AS mk WHERE cct2.kind LIKE '%complete%' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;