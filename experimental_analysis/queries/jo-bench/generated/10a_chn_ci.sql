SELECT * FROM cast_info AS ci, char_name AS chn WHERE ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%' AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;