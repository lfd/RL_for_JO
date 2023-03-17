SELECT * FROM movie_info AS mi, title AS t WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year BETWEEN 1950 AND 2000 AND mi.movie_id = t.id AND t.id = mi.movie_id;