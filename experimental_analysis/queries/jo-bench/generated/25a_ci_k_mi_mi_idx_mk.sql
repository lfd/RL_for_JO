SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info AS mi, cast_info AS ci, movie_info_idx AS mi_idx WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND mi.info = 'Horror' AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;