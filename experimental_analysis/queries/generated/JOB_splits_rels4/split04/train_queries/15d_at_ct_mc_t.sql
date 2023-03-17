SELECT * FROM movie_companies AS mc, title AS t, aka_title AS at, company_type AS ct WHERE t.production_year > 1990 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;