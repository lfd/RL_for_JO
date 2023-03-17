SELECT * FROM company_type AS ct, movie_info_idx AS mi_idx, movie_companies AS mc, title AS t WHERE ct.kind = 'production companies' AND mi_idx.info > '7.0' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;