SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.note IS NULL AND t.production_year = 1998 AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id;