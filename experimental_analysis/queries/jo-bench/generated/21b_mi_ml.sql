SELECT * FROM movie_info AS mi, movie_link AS ml WHERE mi.info IN ('Germany', 'German') AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id;