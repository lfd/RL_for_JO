SELECT * FROM movie_companies AS mc, movie_info AS mi WHERE mc.note LIKE '%(Blu-ray)%' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;