SELECT * FROM info_type AS it, movie_info AS mi, title AS t WHERE mi.info IN ('USA', 'America') AND t.production_year > 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;