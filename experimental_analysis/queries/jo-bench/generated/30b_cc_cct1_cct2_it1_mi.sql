SELECT * FROM info_type AS it1, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1, movie_info AS mi WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete+verified' AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;