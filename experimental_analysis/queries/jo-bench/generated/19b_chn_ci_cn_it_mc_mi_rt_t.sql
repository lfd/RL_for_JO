SELECT * FROM title AS t, cast_info AS ci, char_name AS chn, movie_info AS mi, info_type AS it, role_type AS rt, company_name AS cn, movie_companies AS mc WHERE ci.note = '(voice)' AND cn.country_code = '[us]' AND it.info = 'release dates' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND rt.role = 'actress' AND t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;