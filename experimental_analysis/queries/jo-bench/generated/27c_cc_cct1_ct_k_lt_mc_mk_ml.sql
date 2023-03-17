SELECT * FROM company_type AS ct, comp_cast_type AS cct1, link_type AS lt, keyword AS k, movie_keyword AS mk, movie_companies AS mc, movie_link AS ml, complete_cast AS cc WHERE cct1.kind = 'cast' AND ct.kind = 'production companies' AND k.keyword = 'sequel' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;