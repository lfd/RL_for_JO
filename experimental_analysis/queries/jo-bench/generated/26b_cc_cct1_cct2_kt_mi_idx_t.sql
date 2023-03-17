SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, title AS t, kind_type AS kt, movie_info_idx AS mi_idx WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND kt.kind = 'movie' AND mi_idx.info > '8.0' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;