SELECT * FROM info_type AS it, person_info AS pi WHERE it.info = 'mini biography' AND pi.note IS NOT NULL AND it.id = pi.info_type_id AND pi.info_type_id = it.id;