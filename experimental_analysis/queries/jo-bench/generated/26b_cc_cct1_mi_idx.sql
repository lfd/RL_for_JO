SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND mi_idx.info > '8.0' AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;