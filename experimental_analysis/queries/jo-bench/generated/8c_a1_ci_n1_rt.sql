SELECT * FROM name AS n1, cast_info AS ci, role_type AS rt, aka_name AS a1 WHERE rt.role = 'writer' AND a1.person_id = n1.id AND n1.id = a1.person_id AND n1.id = ci.person_id AND ci.person_id = n1.id AND ci.role_id = rt.id AND rt.id = ci.role_id AND a1.person_id = ci.person_id AND ci.person_id = a1.person_id;