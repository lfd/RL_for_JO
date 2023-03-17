SELECT * FROM info_type AS it2, company_name AS cn2, movie_info_idx AS mi_idx2, movie_companies AS mc2 WHERE it2.info = 'rating' AND mi_idx2.info < '3.5' AND it2.id = mi_idx2.info_type_id AND mi_idx2.info_type_id = it2.id AND cn2.id = mc2.company_id AND mc2.company_id = cn2.id AND mi_idx2.movie_id = mc2.movie_id AND mc2.movie_id = mi_idx2.movie_id;