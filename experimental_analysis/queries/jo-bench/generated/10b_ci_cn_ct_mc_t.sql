SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, company_type AS ct, cast_info AS ci WHERE ci.note LIKE '%(producer)%' AND cn.country_code = '[ru]' AND t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;