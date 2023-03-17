SELECT * FROM role_type AS rt, cast_info AS ci, person_info AS pi, info_type AS it, movie_info AS mi WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND rt.role = 'actress' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id;