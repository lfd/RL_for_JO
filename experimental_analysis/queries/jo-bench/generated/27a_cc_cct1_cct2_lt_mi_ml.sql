SELECT * FROM movie_link AS ml, link_type AS lt, comp_cast_type AS cct1, complete_cast AS cc, comp_cast_type AS cct2, movie_info AS mi WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND lt.link LIKE '%follow%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;