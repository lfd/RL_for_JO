SELECT * FROM cast_info AS ci, name AS n1 WHERE n1.id = ci.person_id AND ci.person_id = n1.id;