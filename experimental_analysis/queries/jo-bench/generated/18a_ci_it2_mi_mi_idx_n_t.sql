SELECT * FROM info_type AS it2, title AS t, movie_info_idx AS mi_idx, movie_info AS mi, cast_info AS ci, name AS n WHERE ci.note IN ('(producer)', '(executive producer)') AND it2.info = 'votes' AND n.gender = 'm' AND n.name LIKE '%Tim%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;