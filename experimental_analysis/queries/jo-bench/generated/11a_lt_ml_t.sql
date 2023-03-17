SELECT * FROM title AS t, movie_link AS ml, link_type AS lt WHERE lt.link LIKE '%follow%' AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id;