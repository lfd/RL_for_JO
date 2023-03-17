SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, link_type AS lt, movie_link AS ml, title AS t, movie_keyword AS mk WHERE cct2.kind LIKE 'complete%' AND lt.link LIKE '%follow%' AND t.production_year BETWEEN 1950 AND 2010 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;