SELECT * FROM movie_info AS mi, complete_cast AS cc, comp_cast_type AS cct1, info_type AS it1, movie_keyword AS mk WHERE cct1.kind IN ('cast', 'crew') AND it1.info = 'genres' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;