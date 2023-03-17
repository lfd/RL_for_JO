SELECT * FROM movie_companies AS mc, company_name AS cn, company_type AS ct, title AS t WHERE cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2008 AND t.id = mc.movie_id AND mc.movie_id = t.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;