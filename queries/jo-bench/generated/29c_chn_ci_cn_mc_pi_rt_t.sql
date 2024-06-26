SELECT * FROM person_info AS pi, role_type AS rt, title AS t, movie_companies AS mc, company_name AS cn, char_name AS chn, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND rt.role = 'actress' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id;