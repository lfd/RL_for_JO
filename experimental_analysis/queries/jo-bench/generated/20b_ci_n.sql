SELECT * FROM cast_info AS ci, name AS n WHERE n.name LIKE '%Downey%Robert%' AND n.id = ci.person_id AND ci.person_id = n.id;