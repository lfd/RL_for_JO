SELECT * FROM cast_info AS ci, movie_companies AS mc, company_name AS cn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id;