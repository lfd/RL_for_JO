SELECT * FROM info_type AS it, movie_info AS mi WHERE it.info = 'release dates' AND it.id = mi.info_type_id AND mi.info_type_id = it.id;