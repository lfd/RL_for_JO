SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1, title AS t WHERE cct1.kind = 'cast' AND t.production_year > 2000 AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;