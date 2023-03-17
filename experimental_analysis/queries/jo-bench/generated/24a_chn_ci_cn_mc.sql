SELECT * FROM company_name AS cn, char_name AS chn, movie_companies AS mc, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;