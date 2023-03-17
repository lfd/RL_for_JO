SELECT * FROM complete_cast AS cc, comp_cast_type AS cct2, info_type AS it2, movie_info_idx AS mi_idx, title AS t, movie_info AS mi WHERE cct2.kind = 'complete+verified' AND it2.info = 'votes' AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;