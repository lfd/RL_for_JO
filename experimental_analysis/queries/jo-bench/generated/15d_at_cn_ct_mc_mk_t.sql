SELECT * FROM movie_keyword AS mk, title AS t, aka_title AS at, company_name AS cn, movie_companies AS mc, company_type AS ct WHERE cn.country_code = '[us]' AND t.production_year > 1990 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;