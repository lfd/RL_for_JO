SELECT * FROM comp_cast_type AS cct2, cast_info AS ci, complete_cast AS cc, title AS t, movie_info AS mi, info_type AS it1 WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;