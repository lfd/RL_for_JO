SELECT * FROM info_type AS it1, title AS t, movie_info AS mi WHERE it1.info = 'budget' AND t.production_year > 2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id;