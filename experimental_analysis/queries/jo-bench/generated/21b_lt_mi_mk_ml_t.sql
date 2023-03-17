SELECT * FROM link_type AS lt, movie_keyword AS mk, title AS t, movie_info AS mi, movie_link AS ml WHERE lt.link LIKE '%follow%' AND mi.info IN ('Germany', 'German') AND t.production_year BETWEEN 2000 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = t.id AND t.id = mi.movie_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;