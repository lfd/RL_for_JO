SELECT * FROM title AS t, movie_companies AS mc, company_name AS cn, cast_info AS ci, char_name AS chn, aka_name AS an WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;