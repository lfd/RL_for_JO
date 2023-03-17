SELECT * FROM info_type AS it2, comp_cast_type AS cct2, complete_cast AS cc, comp_cast_type AS cct1, movie_companies AS mc, movie_info_idx AS mi_idx, company_name AS cn, company_type AS ct, title AS t, kind_type AS kt WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND cn.country_code != '[us]' AND it2.info = 'rating' AND kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info > '6.5' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;