SELECT * FROM info_type AS it, movie_info AS mi, complete_cast AS cc, comp_cast_type AS cct2 WHERE cct2.kind = 'complete+verified' AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;