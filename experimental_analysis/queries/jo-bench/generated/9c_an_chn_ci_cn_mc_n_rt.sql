SELECT * FROM aka_name AS an, name AS n, cast_info AS ci, role_type AS rt, char_name AS chn, movie_companies AS mc, company_name AS cn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;