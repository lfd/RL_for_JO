SELECT * FROM info_type AS it2, movie_info AS mi, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[de]' AND it2.info = 'release dates' AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;