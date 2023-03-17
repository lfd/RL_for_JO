SELECT * FROM title AS t, role_type AS rt, cast_info AS ci, char_name AS chn, company_type AS ct, movie_companies AS mc WHERE ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%' AND rt.role = 'actor' AND t.production_year > 2005 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;