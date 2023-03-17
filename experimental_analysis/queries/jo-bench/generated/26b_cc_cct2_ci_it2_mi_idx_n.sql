SELECT * FROM cast_info AS ci, name AS n, movie_info_idx AS mi_idx, info_type AS it2, complete_cast AS cc, comp_cast_type AS cct2 WHERE cct2.kind LIKE '%complete%' AND it2.info = 'rating' AND mi_idx.info > '8.0' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;