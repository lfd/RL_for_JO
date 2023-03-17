SELECT * FROM info_type AS it1, movie_info AS mi, movie_companies AS mc, title AS t, company_type AS ct WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND t.production_year > 1990 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;