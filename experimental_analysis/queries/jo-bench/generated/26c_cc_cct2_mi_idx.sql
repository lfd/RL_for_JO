SELECT * FROM movie_info_idx AS mi_idx, comp_cast_type AS cct2, complete_cast AS cc WHERE cct2.kind LIKE '%complete%' AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;