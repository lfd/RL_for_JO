SELECT * FROM cast_info AS ci, aka_name AS an, name AS n WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND n.id = ci.person_id AND ci.person_id = n.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;