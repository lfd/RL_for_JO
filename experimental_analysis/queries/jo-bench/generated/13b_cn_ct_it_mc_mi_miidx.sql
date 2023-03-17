SELECT * FROM info_type AS it, company_type AS ct, movie_companies AS mc, company_name AS cn, movie_info_idx AS miidx, movie_info AS mi WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND it.info = 'rating' AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND mi.movie_id = miidx.movie_id AND miidx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;