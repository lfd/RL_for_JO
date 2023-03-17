SELECT * FROM company_type AS ct, movie_info AS mi, movie_companies AS mc WHERE ct.kind = 'production companies' AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;