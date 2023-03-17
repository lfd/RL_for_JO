SELECT * FROM movie_link AS ml, title AS t, link_type AS lt, movie_info AS mi, movie_companies AS mc WHERE lt.link LIKE '%follow%' AND mc.note IS NULL AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = t.id AND t.id = mi.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;