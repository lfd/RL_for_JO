SELECT * FROM company_type AS ct, complete_cast AS cc, link_type AS lt, movie_link AS ml, comp_cast_type AS cct2, movie_companies AS mc WHERE cct2.kind = 'complete' AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;