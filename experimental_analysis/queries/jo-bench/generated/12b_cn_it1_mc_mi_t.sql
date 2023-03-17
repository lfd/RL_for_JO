SELECT * FROM info_type AS it1, title AS t, movie_info AS mi, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND it1.info = 'budget' AND t.production_year > 2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;