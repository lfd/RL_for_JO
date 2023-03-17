SELECT * FROM company_name AS cn, movie_companies AS mc, cast_info AS ci, aka_name AS an WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;