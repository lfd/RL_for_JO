SELECT * FROM link_type AS lt, movie_info AS mi, movie_link AS ml WHERE lt.link LIKE '%follow%' AND mi.info IN ('Germany', 'German') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id;