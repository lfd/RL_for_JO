SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind IN ('cast', 'crew') AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;