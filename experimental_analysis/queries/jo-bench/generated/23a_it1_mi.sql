SELECT * FROM movie_info AS mi, info_type AS it1 WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;