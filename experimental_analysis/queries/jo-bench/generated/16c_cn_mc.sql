SELECT * FROM company_name AS cn, movie_companies AS mc WHERE cn.country_code = '[us]' AND mc.company_id = cn.id AND cn.id = mc.company_id;