SELECT * FROM movie_info AS mi, title AS t WHERE mi.info = 'Horror' AND t.id = mi.movie_id AND mi.movie_id = t.id;