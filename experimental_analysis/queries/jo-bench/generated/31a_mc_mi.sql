SELECT * FROM movie_info AS mi, movie_companies AS mc WHERE mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;