SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, movie_companies AS mc, movie_info AS mi, company_name AS cn WHERE cn.country_code = '[us]' AND it2.info = 'rating' AND mi.info IN ('Drama', 'Horror', 'Western', 'Family') AND mi_idx.info > '7.0' AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;