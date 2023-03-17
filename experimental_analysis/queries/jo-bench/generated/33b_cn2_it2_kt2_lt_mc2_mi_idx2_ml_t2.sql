SELECT * FROM kind_type AS kt2, title AS t2, info_type AS it2, movie_info_idx AS mi_idx2, movie_link AS ml, link_type AS lt, movie_companies AS mc2, company_name AS cn2 WHERE it2.info = 'rating' AND kt2.kind IN ('tv series') AND lt.link LIKE '%follow%' AND mi_idx2.info < '3.0' AND t2.production_year = 2007 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t2.id = ml.linked_movie_id AND ml.linked_movie_id = t2.id AND it2.id = mi_idx2.info_type_id AND mi_idx2.info_type_id = it2.id AND t2.id = mi_idx2.movie_id AND mi_idx2.movie_id = t2.id AND kt2.id = t2.kind_id AND t2.kind_id = kt2.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND t2.id = mc2.movie_id AND mc2.movie_id = t2.id AND ml.linked_movie_id = mi_idx2.movie_id AND mi_idx2.movie_id = ml.linked_movie_id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id AND mi_idx2.movie_id = mc2.movie_id AND mc2.movie_id = mi_idx2.movie_id;