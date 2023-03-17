SELECT * FROM info_type AS it2, movie_info_idx AS mi_idx, movie_companies AS mc, company_type AS ct WHERE it2.info = 'rating' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '8.5' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;