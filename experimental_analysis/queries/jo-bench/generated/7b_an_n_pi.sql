SELECT * FROM aka_name AS an, name AS n, person_info AS pi WHERE an.name LIKE '%a%' AND n.name_pcode_cf LIKE 'D%' AND n.gender = 'm' AND pi.note = 'Volker Boehm' AND n.id = an.person_id AND an.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id;