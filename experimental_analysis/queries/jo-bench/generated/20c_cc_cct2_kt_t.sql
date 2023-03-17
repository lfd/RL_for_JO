SELECT * FROM kind_type AS kt, title AS t, complete_cast AS cc, comp_cast_type AS cct2 WHERE cct2.kind LIKE '%complete%' AND kt.kind = 'movie' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;