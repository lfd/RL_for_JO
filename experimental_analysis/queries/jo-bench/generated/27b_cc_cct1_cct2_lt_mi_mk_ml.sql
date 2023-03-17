SELECT * FROM link_type AS lt, movie_link AS ml, complete_cast AS cc, movie_info AS mi, comp_cast_type AS cct1, comp_cast_type AS cct2, movie_keyword AS mk WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND lt.link LIKE '%follow%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;