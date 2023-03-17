SELECT * FROM complete_cast AS cc, link_type AS lt, movie_link AS ml, movie_info AS mi WHERE lt.link LIKE '%follow%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;