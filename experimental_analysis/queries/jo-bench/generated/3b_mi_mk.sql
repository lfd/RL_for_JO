SELECT * FROM movie_info AS mi, movie_keyword AS mk WHERE mi.info IN ('Bulgaria') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;