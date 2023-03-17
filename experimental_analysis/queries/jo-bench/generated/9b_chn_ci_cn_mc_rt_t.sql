SELECT * FROM cast_info AS ci, char_name AS chn, role_type AS rt, title AS t, company_name AS cn, movie_companies AS mc WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND rt.role = 'actress' AND t.production_year BETWEEN 2007 AND 2010 AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;