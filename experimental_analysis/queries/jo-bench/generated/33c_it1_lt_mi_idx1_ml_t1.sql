SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, info_type AS it1, title AS t1 WHERE it1.info = 'rating' AND lt.link IN ('sequel', 'follows', 'followed by') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id;