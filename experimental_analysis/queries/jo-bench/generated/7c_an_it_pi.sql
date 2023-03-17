SELECT * FROM info_type AS it, person_info AS pi, aka_name AS an WHERE an.name IS NOT NULL AND (an.name LIKE '%a%' OR an.name LIKE 'A%') AND it.info = 'mini biography' AND pi.note IS NOT NULL AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id;