SELECT * FROM person_info AS pi, cast_info AS ci, aka_name AS an WHERE an.name LIKE '%a%' AND pi.note = 'Volker Boehm' AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;