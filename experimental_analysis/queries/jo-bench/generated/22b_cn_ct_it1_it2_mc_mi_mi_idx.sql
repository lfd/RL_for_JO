SELECT * FROM info_type AS it1, company_name AS cn, company_type AS ct, movie_companies AS mc, movie_info_idx AS mi_idx, movie_info AS mi, info_type AS it2 WHERE cn.country_code != '[us]' AND it1.info = 'countries' AND it2.info = 'rating' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND mi_idx.info < '7.0' AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;