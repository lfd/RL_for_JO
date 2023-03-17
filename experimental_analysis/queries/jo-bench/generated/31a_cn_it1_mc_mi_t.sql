SELECT * FROM title AS t, company_name AS cn, movie_companies AS mc, movie_info AS mi, info_type AS it1 WHERE cn.name LIKE 'Lionsgate%' AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id;