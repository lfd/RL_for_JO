SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, movie_companies AS mc, movie_info_idx AS mi_idx, company_type AS ct, company_name AS cn WHERE cct2.kind = 'complete' AND cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '8.5' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;