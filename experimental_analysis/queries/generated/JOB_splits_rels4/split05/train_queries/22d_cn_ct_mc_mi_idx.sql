SELECT * FROM company_type AS ct, movie_companies AS mc, movie_info_idx AS mi_idx, company_name AS cn WHERE cn.country_code != '[us]' AND mi_idx.info < '8.5' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;