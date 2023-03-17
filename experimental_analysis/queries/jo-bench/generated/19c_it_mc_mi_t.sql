SELECT * FROM info_type AS it, title AS t, movie_companies AS mc, movie_info AS mi WHERE it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;