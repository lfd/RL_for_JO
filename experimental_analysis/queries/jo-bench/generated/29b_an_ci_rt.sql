SELECT * FROM role_type AS rt, cast_info AS ci, aka_name AS an WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;