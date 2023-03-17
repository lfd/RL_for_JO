SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1, info_type AS it2, cast_info AS ci, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND it2.info = 'rating' AND mi_idx.info > '8.0' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;