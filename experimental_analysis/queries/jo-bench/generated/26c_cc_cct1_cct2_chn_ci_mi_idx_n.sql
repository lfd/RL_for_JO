SELECT * FROM name AS n, complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, cast_info AS ci, char_name AS chn, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;