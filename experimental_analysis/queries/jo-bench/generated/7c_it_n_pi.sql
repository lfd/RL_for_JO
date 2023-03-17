SELECT * FROM name AS n, person_info AS pi, info_type AS it WHERE it.info = 'mini biography' AND n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender = 'm') AND pi.note IS NOT NULL AND n.id = pi.person_id AND pi.person_id = n.id AND it.id = pi.info_type_id AND pi.info_type_id = it.id;