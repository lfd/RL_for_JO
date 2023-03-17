SELECT * FROM movie_info_idx AS mi_idx, movie_companies AS mc, company_name AS cn, company_type AS ct, title AS t WHERE cn.country_code = '[us]' AND ct.kind IS NOT NULL AND (ct.kind = 'production companies' OR ct.kind = 'distributors') AND t.production_year > 2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;