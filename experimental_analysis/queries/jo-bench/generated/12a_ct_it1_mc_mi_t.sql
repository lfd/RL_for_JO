SELECT * FROM title AS t, info_type AS it1, movie_info AS mi, movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND it1.info = 'genres' AND mi.info IN ('Drama', 'Horror') AND t.production_year BETWEEN 2005 AND 2008 AND t.id = mi.movie_id AND mi.movie_id = t.id AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;