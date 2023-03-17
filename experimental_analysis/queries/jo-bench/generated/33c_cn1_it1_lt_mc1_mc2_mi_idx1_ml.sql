SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, info_type AS it1, movie_companies AS mc1, company_name AS cn1, movie_companies AS mc2 WHERE cn1.country_code != '[us]' AND it1.info = 'rating' AND lt.link IN ('sequel', 'follows', 'followed by') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND cn1.id = mc1.company_id AND mc1.company_id = cn1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;