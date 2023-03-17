SELECT * FROM info_type AS it2, complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND it2.info = 'rating' AND mi_idx.info > '7.0' AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;