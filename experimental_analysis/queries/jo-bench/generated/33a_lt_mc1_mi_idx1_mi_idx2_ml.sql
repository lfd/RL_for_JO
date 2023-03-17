SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx2, movie_companies AS mc1 WHERE lt.link IN ('sequel', 'follows', 'followed by') AND mi_idx2.info < '3.0' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id;