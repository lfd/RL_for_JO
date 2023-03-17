SELECT * FROM company_name AS cn, movie_companies AS mc, movie_info AS mi, info_type AS it1 WHERE cn.name LIKE 'Lionsgate%' AND it1.info = 'genres' AND mc.note LIKE '%(Blu-ray)%' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id;