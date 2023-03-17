SELECT * FROM info_type AS it, movie_info AS mi, company_name AS cn, movie_companies AS mc WHERE cn.country_code = '[us]' AND cn.name = 'DreamWorks Animation' AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;