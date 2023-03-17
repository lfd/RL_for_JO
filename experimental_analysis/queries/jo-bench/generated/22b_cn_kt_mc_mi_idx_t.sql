SELECT * FROM kind_type AS kt, movie_companies AS mc, title AS t, company_name AS cn, movie_info_idx AS mi_idx WHERE cn.country_code != '[us]' AND kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '7.0' AND t.production_year > 2009 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;