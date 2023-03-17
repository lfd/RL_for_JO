SELECT * FROM title AS t, complete_cast AS cc, movie_info AS mi, info_type AS it1 WHERE it1.info = 'release dates' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 1990 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;