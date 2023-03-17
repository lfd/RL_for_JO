SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, cast_info AS ci, role_type AS rt, name AS n WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id;