SELECT * FROM role_type AS rt, aka_name AS an1, cast_info AS ci WHERE ci.note = '(voice: English version)' AND rt.role = 'actress' AND ci.role_id = rt.id AND rt.id = ci.role_id AND an1.person_id = ci.person_id AND ci.person_id = an1.person_id;