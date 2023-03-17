SELECT * FROM company_type AS ct, movie_companies AS mc, company_name AS cn, movie_info_idx AS miidx WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND miidx.movie_id = mc.movie_id AND mc.movie_id = miidx.movie_id;