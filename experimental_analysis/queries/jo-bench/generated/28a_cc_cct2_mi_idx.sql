SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, movie_info_idx AS mi_idx WHERE cct2.kind != 'complete+verified' AND mi_idx.info < '8.5' AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;