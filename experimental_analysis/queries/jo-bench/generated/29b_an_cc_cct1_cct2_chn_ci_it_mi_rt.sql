SELECT * FROM info_type AS it, movie_info AS mi, complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, char_name AS chn, cast_info AS ci, aka_name AS an, role_type AS rt WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.info LIKE 'USA:%200%' AND rt.role = 'actress' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;