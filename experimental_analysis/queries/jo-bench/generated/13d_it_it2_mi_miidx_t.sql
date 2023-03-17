SELECT * FROM movie_info AS mi, info_type AS it, movie_info_idx AS miidx, title AS t, info_type AS it2 WHERE it.info = 'rating' AND it2.info = 'release dates' AND mi.movie_id = t.id AND t.id = mi.movie_id AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id;