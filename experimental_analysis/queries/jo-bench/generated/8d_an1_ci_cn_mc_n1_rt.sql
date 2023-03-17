SELECT * FROM movie_companies AS mc, company_name AS cn, cast_info AS ci, role_type AS rt, name AS n1, aka_name AS an1 WHERE cn.country_code = '[us]' AND rt.role = 'costume designer' AND an1.person_id = n1.id AND n1.id = an1.person_id AND n1.id = ci.person_id AND ci.person_id = n1.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND an1.person_id = ci.person_id AND ci.person_id = an1.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;