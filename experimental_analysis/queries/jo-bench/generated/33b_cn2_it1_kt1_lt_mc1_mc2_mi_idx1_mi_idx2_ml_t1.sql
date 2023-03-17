SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, info_type AS it1, title AS t1, kind_type AS kt1, movie_companies AS mc2, movie_info_idx AS mi_idx2, company_name AS cn2, movie_companies AS mc1 WHERE it1.info = 'rating' AND kt1.kind IN ('tv series') AND lt.link LIKE '%follow%' AND mi_idx2.info < '3.0' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id AND t1.id = mc1.movie_id AND mc1.movie_id = t1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id AND mi_idx2.movie_id = mc2.movie_id AND mc2.movie_id = mi_idx2.movie_id;