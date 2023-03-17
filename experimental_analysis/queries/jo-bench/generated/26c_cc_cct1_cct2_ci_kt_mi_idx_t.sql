SELECT * FROM kind_type AS kt, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1, title AS t, cast_info AS ci, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND kt.kind = 'movie' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;