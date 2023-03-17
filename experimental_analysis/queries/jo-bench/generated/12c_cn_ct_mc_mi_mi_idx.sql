SELECT * FROM movie_info AS mi, movie_companies AS mc, movie_info_idx AS mi_idx, company_name AS cn, company_type AS ct WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND mi.info IN ('Drama', 'Horror', 'Western', 'Family') AND mi_idx.info > '7.0' AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;