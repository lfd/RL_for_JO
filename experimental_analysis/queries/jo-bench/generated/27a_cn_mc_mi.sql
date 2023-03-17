SELECT * FROM movie_info AS mi, movie_companies AS mc, company_name AS cn WHERE cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.note IS NULL AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mc.company_id = cn.id AND cn.id = mc.company_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;