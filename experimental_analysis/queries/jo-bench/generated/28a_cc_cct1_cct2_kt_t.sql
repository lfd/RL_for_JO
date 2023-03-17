SELECT * FROM comp_cast_type AS cct1, comp_cast_type AS cct2, complete_cast AS cc, title AS t, kind_type AS kt WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND kt.kind IN ('movie', 'episode') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;