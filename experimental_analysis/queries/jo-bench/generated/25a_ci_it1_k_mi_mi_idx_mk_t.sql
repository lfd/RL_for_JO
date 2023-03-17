SELECT * FROM title AS t, info_type AS it1, movie_info_idx AS mi_idx, movie_keyword AS mk, movie_info AS mi, keyword AS k, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND mi.info = 'Horror' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;