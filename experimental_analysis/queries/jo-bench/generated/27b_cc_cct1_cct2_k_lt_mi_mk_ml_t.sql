SELECT * FROM keyword AS k, comp_cast_type AS cct2, complete_cast AS cc, link_type AS lt, movie_link AS ml, movie_info AS mi, title AS t, movie_keyword AS mk, comp_cast_type AS cct1 WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND k.keyword = 'sequel' AND lt.link LIKE '%follow%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND t.production_year = 1998 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mi.movie_id = t.id AND t.id = mi.movie_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;