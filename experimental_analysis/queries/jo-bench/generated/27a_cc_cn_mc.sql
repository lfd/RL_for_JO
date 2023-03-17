SELECT * FROM complete_cast AS cc, movie_companies AS mc, company_name AS cn WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.note IS NULL AND mc.company_id = cn.id AND cn.id = mc.company_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;