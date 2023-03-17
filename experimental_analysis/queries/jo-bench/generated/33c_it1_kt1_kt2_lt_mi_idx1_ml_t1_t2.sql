SELECT * FROM kind_type AS kt2, link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx1, info_type AS it1, title AS t1, kind_type AS kt1, title AS t2 WHERE it1.info = 'rating' AND kt1.kind IN ('tv series', 'episode') AND kt2.kind IN ('tv series', 'episode') AND lt.link IN ('sequel', 'follows', 'followed by') AND t2.production_year BETWEEN 2000 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id;