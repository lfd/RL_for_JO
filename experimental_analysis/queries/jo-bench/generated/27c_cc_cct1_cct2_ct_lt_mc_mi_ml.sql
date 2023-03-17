SELECT * FROM company_type AS ct, comp_cast_type AS cct1, complete_cast AS cc, link_type AS lt, movie_link AS ml, comp_cast_type AS cct2, movie_companies AS mc, movie_info AS mi WHERE cct1.kind = 'cast' AND cct2.kind LIKE 'complete%' AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'English') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;