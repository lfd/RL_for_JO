SELECT * FROM link_type AS lt, movie_link AS ml, movie_info_idx AS mi_idx1, movie_companies AS mc2, company_name AS cn2, info_type AS it1 WHERE it1.info = 'rating' AND lt.link LIKE '%follow%' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;