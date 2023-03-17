SELECT * FROM info_type AS it1, movie_info AS mi, title AS t, movie_companies AS mc, company_type AS ct, company_name AS cn WHERE cn.country_code = '[us]' AND ct.kind IS NOT NULL AND (ct.kind = 'production companies' OR ct.kind = 'distributors') AND it1.info = 'budget' AND t.production_year > 2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;