SELECT * FROM person_info AS pi, cast_info AS ci WHERE pi.note = 'Volker Boehm' AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id;