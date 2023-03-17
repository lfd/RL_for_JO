SELECT * FROM movie_companies AS mc2, link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx2 WHERE lt.link IN ('sequel', 'follows', 'followed by') AND mi_idx2.info < '3.5' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id AND mi_idx2.movie_id = mc2.movie_id AND mc2.movie_id = mi_idx2.movie_id;