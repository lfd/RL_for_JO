SELECT * FROM name AS n, cast_info AS ci, char_name AS chn WHERE ci.note = '(voice)' AND n.gender = 'f' AND n.name LIKE '%Angel%' AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;