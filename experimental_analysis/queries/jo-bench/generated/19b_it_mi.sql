SELECT * FROM info_type AS it, movie_info AS mi WHERE it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND it.id = mi.info_type_id AND mi.info_type_id = it.id;