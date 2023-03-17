SELECT * FROM info_type AS it2, cast_info AS ci, name AS n, movie_info_idx AS mi_idx, char_name AS chn, title AS t, kind_type AS kt WHERE chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND it2.info = 'rating' AND kt.kind = 'movie' AND mi_idx.info > '8.0' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;