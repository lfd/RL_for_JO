SELECT * FROM company_name AS cn, movie_companies AS mc, cast_info AS ci, role_type AS rt, company_type AS ct WHERE ci.note LIKE '%(producer)%' AND cn.country_code = '[us]' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;