SELECT * FROM name AS n, cast_info AS ci, char_name AS chn WHERE chn.name NOT LIKE '%Sherlock%' AND (chn.name LIKE '%Tony%Stark%' OR chn.name LIKE '%Iron%Man%') AND n.name LIKE '%Downey%Robert%' AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id;