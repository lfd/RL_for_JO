SELECT * FROM info_type AS it, movie_info AS mi, movie_companies AS mc, title AS t WHERE mc.note NOT LIKE '%(TV)%' AND mc.note LIKE '%(USA)%' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 1990 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;