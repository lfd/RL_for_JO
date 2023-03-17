SELECT * FROM info_type AS it, person_info AS pi, aka_name AS an, name AS n WHERE an.name IS NOT NULL AND (an.name LIKE '%a%' OR an.name LIKE 'A%') AND it.info = 'mini biography' AND n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND pi.note IS NOT NULL AND n.id = an.person_id AND an.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id;