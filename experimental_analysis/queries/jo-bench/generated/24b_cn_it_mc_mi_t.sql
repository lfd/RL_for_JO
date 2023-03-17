SELECT * FROM company_name AS cn, movie_info AS mi, movie_companies AS mc, title AS t, info_type AS it WHERE cn.country_code = '[us]' AND cn.name = 'DreamWorks Animation' AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id;