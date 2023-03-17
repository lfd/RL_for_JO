SELECT * FROM cast_info AS ci, complete_cast AS cc, comp_cast_type AS cct1, movie_info_idx AS mi_idx, movie_keyword AS mk WHERE cct1.kind IN ('cast', 'crew') AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;