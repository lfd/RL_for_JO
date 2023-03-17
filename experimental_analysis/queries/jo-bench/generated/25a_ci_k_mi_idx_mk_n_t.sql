SELECT * FROM name AS n, movie_keyword AS mk, cast_info AS ci, title AS t, movie_info_idx AS mi_idx, keyword AS k WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND n.gender = 'm' AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;