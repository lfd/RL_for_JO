SELECT * FROM movie_info AS mi, title AS t, info_type AS it2 WHERE it2.info = 'release dates' AND t.title != '' AND (t.title LIKE '%Champion%' OR t.title LIKE '%Loser%') AND mi.movie_id = t.id AND t.id = mi.movie_id AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id;