SELECT * FROM cast_info AS ci, info_type AS it2, complete_cast AS cc, comp_cast_type AS cct1, movie_info_idx AS mi_idx, comp_cast_type AS cct2, title AS t, char_name AS chn WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND it2.info = 'rating' AND t.production_year > 2000 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;