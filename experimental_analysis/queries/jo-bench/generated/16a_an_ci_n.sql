SELECT * FROM name AS n, cast_info AS ci, aka_name AS an WHERE an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;