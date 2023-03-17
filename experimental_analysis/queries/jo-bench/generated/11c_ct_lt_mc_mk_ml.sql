SELECT * FROM link_type AS lt, movie_link AS ml, movie_companies AS mc, company_type AS ct, movie_keyword AS mk WHERE ct.kind != 'production companies' AND ct.kind IS NOT NULL AND mc.note IS NOT NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;