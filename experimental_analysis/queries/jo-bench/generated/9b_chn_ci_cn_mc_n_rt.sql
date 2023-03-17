SELECT * FROM company_name AS cn, movie_companies AS mc, cast_info AS ci, role_type AS rt, char_name AS chn, name AS n WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;