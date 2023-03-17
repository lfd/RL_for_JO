SELECT * FROM info_type AS it1, movie_info AS mi, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;