SELECT * FROM person_info AS pi, aka_name AS an, name AS n, cast_info AS ci WHERE an.name LIKE '%a%' AND n.name_pcode_cf LIKE 'D%' AND n.gender = 'm' AND pi.note = 'Volker Boehm' AND n.id = an.person_id AND an.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = n.id AND n.id = ci.person_id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;