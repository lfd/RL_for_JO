SELECT * FROM aka_name AS a1, cast_info AS ci, role_type AS rt WHERE rt.role = 'writer' AND ci.role_id = rt.id AND rt.id = ci.role_id AND a1.person_id = ci.person_id AND ci.person_id = a1.person_id;