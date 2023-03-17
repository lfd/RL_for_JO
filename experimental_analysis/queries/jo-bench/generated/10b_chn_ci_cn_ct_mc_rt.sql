SELECT * FROM company_name AS cn, movie_companies AS mc, company_type AS ct, cast_info AS ci, char_name AS chn, role_type AS rt WHERE ci.note LIKE '%(producer)%' AND cn.country_code = '[ru]' AND rt.role = 'actor' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;