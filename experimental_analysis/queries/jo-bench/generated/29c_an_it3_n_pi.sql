SELECT * FROM info_type AS it3, name AS n, person_info AS pi, aka_name AS an WHERE it3.info = 'trivia' AND n.gender = 'f' AND n.name LIKE '%An%' AND n.id = an.person_id AND an.person_id = n.id AND n.id = pi.person_id AND pi.person_id = n.id AND it3.id = pi.info_type_id AND pi.info_type_id = it3.id;