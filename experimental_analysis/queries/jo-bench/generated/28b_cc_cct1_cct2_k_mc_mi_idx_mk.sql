SELECT * FROM movie_companies AS mc, comp_cast_type AS cct2, comp_cast_type AS cct1, keyword AS k, movie_keyword AS mk, complete_cast AS cc, movie_info_idx AS mi_idx WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi_idx.info > '6.5' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;