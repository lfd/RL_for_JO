SELECT * FROM movie_info AS mi, cast_info AS ci WHERE ci.note IN ('(producer)', '(executive producer)') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id;