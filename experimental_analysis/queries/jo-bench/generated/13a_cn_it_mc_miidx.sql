SELECT * FROM company_name AS cn, movie_companies AS mc, movie_info_idx AS miidx, info_type AS it WHERE cn.country_code = '[de]' AND it.info = 'rating' AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = miidx.info_type_id AND miidx.info_type_id = it.id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;