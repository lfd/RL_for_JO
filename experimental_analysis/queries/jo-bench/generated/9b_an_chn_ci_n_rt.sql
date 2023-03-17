SELECT * FROM cast_info AS ci, role_type AS rt, aka_name AS an, name AS n, char_name AS chn WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;