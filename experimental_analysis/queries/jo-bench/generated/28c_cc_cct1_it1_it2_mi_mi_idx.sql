SELECT * FROM info_type AS it1, movie_info AS mi, info_type AS it2, movie_info_idx AS mi_idx, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND it1.info = 'countries' AND it2.info = 'rating' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi_idx.info < '8.5' AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;