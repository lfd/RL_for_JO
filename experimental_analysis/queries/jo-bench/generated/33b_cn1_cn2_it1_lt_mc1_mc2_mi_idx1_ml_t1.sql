SELECT * FROM movie_info_idx AS mi_idx1, link_type AS lt, movie_link AS ml, info_type AS it1, movie_companies AS mc1, company_name AS cn1, movie_companies AS mc2, company_name AS cn2, title AS t1 WHERE cn1.country_code = '[nl]' AND it1.info = 'rating' AND lt.link LIKE '%follow%' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND t1.id = ml.movie_id AND ml.movie_id = t1.id AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND cn1.id = mc1.company_id AND mc1.company_id = cn1.id AND t1.id = mc1.movie_id AND mc1.movie_id = t1.id AND ml.movie_id = mi_idx1.movie_id AND mi_idx1.movie_id = ml.movie_id AND ml.movie_id = mc1.movie_id AND mc1.movie_id = ml.movie_id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND ml.linked_movie_id = mc2.movie_id AND mc2.movie_id = ml.linked_movie_id;