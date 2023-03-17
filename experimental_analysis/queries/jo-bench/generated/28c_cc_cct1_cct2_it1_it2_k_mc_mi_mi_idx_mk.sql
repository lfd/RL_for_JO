SELECT * FROM info_type AS it2, movie_companies AS mc, movie_info_idx AS mi_idx, info_type AS it1, movie_info AS mi, comp_cast_type AS cct1, keyword AS k, movie_keyword AS mk, complete_cast AS cc, comp_cast_type AS cct2 WHERE cct1.kind = 'cast' AND cct2.kind = 'complete' AND it1.info = 'countries' AND it2.info = 'rating' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi_idx.info < '8.5' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;