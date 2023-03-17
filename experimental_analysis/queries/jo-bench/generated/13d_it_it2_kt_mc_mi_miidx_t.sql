SELECT * FROM title AS t, kind_type AS kt, movie_companies AS mc, info_type AS it2, info_type AS it, movie_info_idx AS miidx, movie_info AS mi WHERE it.info = 'rating' AND it2.info = 'release dates' AND kt.kind = 'movie' AND mi.movie_id = t.id AND t.id = mi.movie_id AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;