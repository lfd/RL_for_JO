SELECT * FROM movie_info_idx AS miidx, info_type AS it, movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND it.info = 'rating' AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;