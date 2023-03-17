SELECT * FROM info_type AS it, movie_info AS mi, movie_companies AS mc WHERE it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;