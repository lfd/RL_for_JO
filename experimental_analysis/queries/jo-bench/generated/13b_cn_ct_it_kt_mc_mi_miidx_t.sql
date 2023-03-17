SELECT * FROM movie_info_idx AS miidx, info_type AS it, title AS t, kind_type AS kt, movie_companies AS mc, company_type AS ct, company_name AS cn, movie_info AS mi WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND it.info = 'rating' AND kt.kind = 'movie' AND t.title != '' AND (t.title LIKE '%Champion%' OR t.title LIKE '%Loser%') AND mi.movie_id = t.id AND t.id = mi.movie_id AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND miidx.movie_id = t.id AND t.id = miidx.movie_id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;