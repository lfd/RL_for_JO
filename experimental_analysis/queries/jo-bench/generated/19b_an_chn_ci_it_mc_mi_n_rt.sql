SELECT * FROM info_type AS it, aka_name AS an, name AS n, cast_info AS ci, char_name AS chn, role_type AS rt, movie_info AS mi, movie_companies AS mc WHERE ci.note = '(voice)' AND it.info = 'release dates' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%') AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;