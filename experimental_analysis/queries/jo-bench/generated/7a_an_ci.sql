SELECT * FROM aka_name AS an, cast_info AS ci WHERE an.name LIKE '%a%' AND an.person_id = ci.person_id AND ci.person_id = an.person_id;