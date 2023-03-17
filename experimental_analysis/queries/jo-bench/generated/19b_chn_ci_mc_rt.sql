SELECT * FROM char_name AS chn, cast_info AS ci, movie_companies AS mc, role_type AS rt WHERE ci.note = '(voice)' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND rt.role = 'actress' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;