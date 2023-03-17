SELECT * FROM title AS t, comp_cast_type AS cct1, complete_cast AS cc WHERE cct1.kind = 'cast' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;