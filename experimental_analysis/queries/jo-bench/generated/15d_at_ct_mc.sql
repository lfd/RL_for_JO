SELECT * FROM company_type AS ct, movie_companies AS mc, aka_title AS at WHERE mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;