SELECT * FROM title AS t, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'complete+verified' AND t.production_year > 1990 AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;