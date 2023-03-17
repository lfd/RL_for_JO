SELECT * FROM title AS t, comp_cast_type AS cct1, movie_info AS mi, complete_cast AS cc WHERE cct1.kind = 'cast' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;