SELECT * FROM info_type AS it, person_info AS pi, cast_info AS ci WHERE it.info = 'mini biography' AND pi.note IS NOT NULL AND it.id = pi.info_type_id AND pi.info_type_id = it.id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id;