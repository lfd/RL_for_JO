SELECT * FROM movie_info AS mi, title AS t WHERE t.id = mi.movie_id AND mi.movie_id = t.id;