SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, company_type AS ct WHERE cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND t.production_year > 2000 AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;