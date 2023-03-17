SELECT * FROM company_name AS cn, movie_companies AS mc, movie_info AS mi, cast_info AS ci, char_name AS chn WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;