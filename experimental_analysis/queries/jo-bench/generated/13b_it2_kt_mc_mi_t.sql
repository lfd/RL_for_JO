SELECT * FROM info_type AS it2, movie_info AS mi, movie_companies AS mc, kind_type AS kt, title AS t WHERE it2.info = 'release dates' AND kt.kind = 'movie' AND t.title != '' AND (t.title LIKE '%Champion%' OR t.title LIKE '%Loser%') AND mi.movie_id = t.id AND t.id = mi.movie_id AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;