SELECT * FROM movie_keyword AS mk, comp_cast_type AS cct2, complete_cast AS cc, title AS t WHERE cct2.kind = 'complete' AND t.production_year BETWEEN 1950 AND 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;