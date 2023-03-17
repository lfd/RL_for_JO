SELECT * FROM movie_keyword AS mk, title AS t, movie_companies AS mc, company_name AS cn WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;