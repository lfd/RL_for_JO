SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info_idx AS mi_idx, info_type AS it2, cast_info AS ci, name AS n, movie_companies AS mc WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it2.info = 'votes' AND k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mc.note LIKE '%(Blu-ray)%' AND n.gender = 'm' AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;