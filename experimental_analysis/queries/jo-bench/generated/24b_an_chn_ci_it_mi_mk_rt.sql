SELECT * FROM info_type AS it, movie_info AS mi, aka_name AS an, cast_info AS ci, char_name AS chn, movie_keyword AS mk, role_type AS rt WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND rt.role = 'actress' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;