SELECT * FROM movie_info AS mi, cast_info AS ci, movie_keyword AS mk WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id;