SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, aka_title AS at WHERE cn.country_code = '[us]' AND t.production_year > 1990 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;