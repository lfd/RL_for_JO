SELECT * FROM info_type AS it2, movie_info AS mi, movie_companies AS mc, kind_type AS kt, title AS t, company_type AS ct, movie_info_idx AS miidx WHERE ct.kind = 'production companies' AND it2.info = 'release dates' AND kt.kind = 'movie' AND t.title != '' AND (t.title LIKE '%Champion%' OR t.title LIKE '%Loser%') AND mi.movie_id = t.id AND t.id = mi.movie_id AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;