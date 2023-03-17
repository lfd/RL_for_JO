SELECT * FROM company_type AS ct, movie_info AS mi, movie_companies AS mc, company_name AS cn, info_type AS it2 WHERE cn.country_code = '[us]' AND ct.kind = 'production companies' AND it2.info = 'release dates' AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;