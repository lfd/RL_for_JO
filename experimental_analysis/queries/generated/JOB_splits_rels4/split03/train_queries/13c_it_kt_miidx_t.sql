SELECT * FROM movie_info_idx AS miidx, info_type AS it, title AS t, kind_type AS kt WHERE it.info = 'rating' AND kt.kind = 'movie' AND t.title != '' AND (t.title LIKE 'Champion%' OR t.title LIKE 'Loser%') AND kt.id = t.kind_id AND t.kind_id = kt.id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id;