SELECT * FROM keyword AS k, link_type AS lt, movie_link AS ml, complete_cast AS cc, movie_keyword AS mk, title AS t, comp_cast_type AS cct1 WHERE cct1.kind IN ('cast', 'crew') AND k.keyword = 'sequel' AND lt.link LIKE '%follow%' AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;