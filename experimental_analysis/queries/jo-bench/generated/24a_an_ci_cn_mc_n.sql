SELECT * FROM name AS n, aka_name AS an, cast_info AS ci, company_name AS cn, movie_companies AS mc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND n.gender = 'f' AND n.name LIKE '%An%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;