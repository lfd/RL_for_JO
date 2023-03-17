SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, cast_info AS ci, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;