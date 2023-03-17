SELECT * FROM movie_info AS mi, info_type AS it1 WHERE it1.info = 'genres' AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;