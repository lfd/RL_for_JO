SELECT * FROM company_type AS ct, movie_companies AS mc, company_name AS cn WHERE cn.country_code = '[de]' AND ct.kind = 'production companies' AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;