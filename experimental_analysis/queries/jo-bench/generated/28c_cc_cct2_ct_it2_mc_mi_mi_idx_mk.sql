SELECT * FROM movie_companies AS mc, complete_cast AS cc, comp_cast_type AS cct2, movie_keyword AS mk, company_type AS ct, movie_info AS mi, info_type AS it2, movie_info_idx AS mi_idx WHERE cct2.kind = 'complete' AND it2.info = 'rating' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi_idx.info < '8.5' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;