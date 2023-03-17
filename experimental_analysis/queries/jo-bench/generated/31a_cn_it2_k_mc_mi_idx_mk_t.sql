SELECT * FROM movie_info_idx AS mi_idx, info_type AS it2, movie_keyword AS mk, title AS t, company_name AS cn, movie_companies AS mc, keyword AS k WHERE cn.name LIKE 'Lionsgate%' AND it2.info = 'votes' AND k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cn.id = mc.company_id AND mc.company_id = cn.id;