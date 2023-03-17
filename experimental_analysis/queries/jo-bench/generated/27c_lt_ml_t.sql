SELECT * FROM link_type AS lt, movie_link AS ml, title AS t WHERE lt.link LIKE '%follow%' AND t.production_year BETWEEN 1950 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id;