SELECT * FROM link_type AS lt, movie_link AS ml WHERE lt.link = 'features' AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id;