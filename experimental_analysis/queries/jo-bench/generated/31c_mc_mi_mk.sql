SELECT * FROM movie_info AS mi, movie_companies AS mc, movie_keyword AS mk WHERE mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;