SELECT * FROM aka_name AS an, name AS n WHERE n.gender = 'f' AND n.name LIKE '%Angel%' AND n.id = an.person_id AND an.person_id = n.id;