SELECT * FROM movie_keyword AS mk, info_type AS it1, cast_info AS ci, movie_info AS mi WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;