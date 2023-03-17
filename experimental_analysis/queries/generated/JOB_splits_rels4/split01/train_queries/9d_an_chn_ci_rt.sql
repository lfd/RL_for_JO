SELECT * FROM char_name AS chn, cast_info AS ci, role_type AS rt, aka_name AS an WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND rt.role = 'actress' AND ci.role_id = rt.id AND rt.id = ci.role_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;