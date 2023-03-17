SELECT * FROM info_type AS it2, comp_cast_type AS cct2, movie_companies AS mc, company_name AS cn, complete_cast AS cc, movie_info_idx AS mi_idx WHERE cct2.kind != 'complete+verified' AND cn.country_code != '[us]' AND it2.info = 'rating' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '8.5' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;