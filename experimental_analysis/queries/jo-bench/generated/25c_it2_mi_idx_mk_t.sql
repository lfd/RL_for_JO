SELECT * FROM movie_keyword AS mk, movie_info_idx AS mi_idx, info_type AS it2, title AS t WHERE it2.info = 'votes' AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;