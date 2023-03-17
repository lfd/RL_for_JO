SELECT * FROM title AS t, movie_info AS mi, info_type AS it WHERE it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;