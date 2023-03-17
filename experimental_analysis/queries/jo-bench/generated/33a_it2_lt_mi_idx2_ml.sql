SELECT * FROM info_type AS it2, link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx2 WHERE it2.info = 'rating' AND lt.link IN ('sequel', 'follows', 'followed by') AND mi_idx2.info < '3.0' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND it2.id = mi_idx2.info_type_id AND mi_idx2.info_type_id = it2.id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id;