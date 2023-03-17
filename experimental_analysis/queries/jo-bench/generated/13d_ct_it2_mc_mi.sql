SELECT * FROM info_type AS it2, movie_companies AS mc, movie_info AS mi, company_type AS ct WHERE ct.kind = 'production companies' AND it2.info = 'release dates' AND it2.id = mi.info_type_id AND mi.info_type_id = it2.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;