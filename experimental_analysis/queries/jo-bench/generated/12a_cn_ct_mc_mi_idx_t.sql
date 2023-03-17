SELECT * FROM movie_info_idx AS mi_idx, movie_companies AS mc, company_name AS cn, company_type AS ct, title AS t WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND mi_idx.info > '8.0' AND t.production_year BETWEEN 2005 AND 2008 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;