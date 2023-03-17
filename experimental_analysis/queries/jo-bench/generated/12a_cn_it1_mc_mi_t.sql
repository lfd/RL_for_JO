SELECT * FROM info_type AS it1, movie_info AS mi, title AS t, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND it1.info = 'genres' AND mi.info IN ('Drama', 'Horror') AND t.production_year BETWEEN 2005 AND 2008 AND t.id = mi.movie_id AND mi.movie_id = t.id AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;