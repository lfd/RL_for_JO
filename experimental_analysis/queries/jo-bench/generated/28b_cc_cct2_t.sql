SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, title AS t WHERE cct2.kind != 'complete+verified' AND t.production_year > 2005 AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;