SELECT * FROM movie_info AS mi, title AS t WHERE mi.info IN ('Drama', 'Horror', 'Western', 'Family') AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id;