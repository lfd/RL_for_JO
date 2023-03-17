SELECT * FROM movie_info_idx AS mi_idx1, title AS t1, kind_type AS kt1, link_type AS lt, movie_link AS ml WHERE kt1.kind IN ('tv series') AND lt.link LIKE '%follow%' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id;