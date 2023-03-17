SELECT * FROM title AS t, complete_cast AS cc, comp_cast_type AS cct1, movie_info AS mi WHERE cct1.kind = 'complete+verified' AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;