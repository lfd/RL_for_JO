SELECT * FROM cast_info AS ci, name AS n, role_type AS rt WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND rt.role = 'actress' AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id;