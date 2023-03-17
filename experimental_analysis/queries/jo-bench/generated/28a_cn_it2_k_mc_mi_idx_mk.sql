SELECT * FROM movie_companies AS mc, info_type AS it2, keyword AS k, movie_keyword AS mk, movie_info_idx AS mi_idx, company_name AS cn WHERE cn.country_code != '[us]' AND it2.info = 'rating' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info < '8.5' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cn.id = mc.company_id AND mc.company_id = cn.id;