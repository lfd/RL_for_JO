SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, info_type AS it1, movie_info_idx AS mi_idx2, info_type AS it2, movie_companies AS mc1 WHERE it1.info = 'rating' AND it2.info = 'rating' AND lt.link IN ('sequel', 'follows', 'followed by') AND mi_idx2.info < '3.0' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND it2.id = mi_idx2.info_type_id AND mi_idx2.info_type_id = it2.id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id;