SELECT * FROM movie_companies AS mc, title AS t, company_name AS cn, movie_info_idx AS mi_idx WHERE cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '7.0' AND t.production_year > 2008 AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;