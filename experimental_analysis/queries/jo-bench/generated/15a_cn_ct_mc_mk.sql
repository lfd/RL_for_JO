SELECT * FROM company_type AS ct, company_name AS cn, movie_companies AS mc, movie_keyword AS mk WHERE cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;