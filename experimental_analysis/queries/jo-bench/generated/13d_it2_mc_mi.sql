SELECT * FROM info_type AS it2, movie_info AS mi, movie_companies AS mc WHERE it2.info = 'release dates' AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;