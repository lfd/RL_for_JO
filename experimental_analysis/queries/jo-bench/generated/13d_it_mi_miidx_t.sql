SELECT * FROM movie_info AS mi, info_type AS it, movie_info_idx AS miidx, title AS t WHERE it.info = 'rating' AND mi.movie_id = t.id AND t.id = mi.movie_id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id;