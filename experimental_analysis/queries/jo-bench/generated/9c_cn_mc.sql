SELECT * FROM movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[us]' AND mc.company_id = cn.id AND cn.id = mc.company_id;