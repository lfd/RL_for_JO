SELECT * FROM info_type AS it1, link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx1, movie_companies AS mc2, company_name AS cn2, title AS t1, kind_type AS kt1, title AS t2, kind_type AS kt2 WHERE it1.info = 'rating' AND kt1.kind IN ('tv series') AND kt2.kind IN ('tv series') AND lt.link IN ('sequel', 'follows', 'followed by') AND t2.production_year BETWEEN 2005 AND 2008 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND t2.id = mc2.movie_id AND mc2.movie_id = t2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;