SELECT * FROM complete_cast AS cc, comp_cast_type AS cct2, cast_info AS ci, aka_name AS an, role_type AS rt WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;