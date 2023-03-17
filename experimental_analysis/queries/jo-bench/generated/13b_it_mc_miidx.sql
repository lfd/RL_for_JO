SELECT * FROM movie_info_idx AS miidx, info_type AS it, movie_companies AS mc WHERE it.info = 'rating' AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;