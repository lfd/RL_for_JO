SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code = '[ru]' AND t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id;