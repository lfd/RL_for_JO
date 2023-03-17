SELECT * FROM movie_keyword AS mk, company_type AS ct, movie_companies AS mc, title AS t, company_name AS cn, movie_info_idx AS mi_idx WHERE cn.country_code != '[us]' AND mi_idx.info < '8.5' AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;