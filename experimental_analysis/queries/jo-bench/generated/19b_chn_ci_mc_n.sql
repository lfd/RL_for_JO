SELECT * FROM name AS n, cast_info AS ci, movie_companies AS mc, char_name AS chn WHERE ci.note = '(voice)' AND mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND n.gender = 'f' AND n.name LIKE '%Angel%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;