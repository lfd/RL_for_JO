SELECT * FROM kind_type AS kt, title AS t, complete_cast AS cc, comp_cast_type AS cct1, cast_info AS ci WHERE cct1.kind = 'cast' AND kt.kind = 'movie' AND t.production_year > 1950 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;