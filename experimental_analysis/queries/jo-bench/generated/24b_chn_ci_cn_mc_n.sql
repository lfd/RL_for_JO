SELECT * FROM name AS n, cast_info AS ci, char_name AS chn, movie_companies AS mc, company_name AS cn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND cn.name = 'DreamWorks Animation' AND n.gender = 'f' AND n.name LIKE '%An%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;