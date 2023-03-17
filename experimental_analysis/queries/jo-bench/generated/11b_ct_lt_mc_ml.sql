SELECT * FROM link_type AS lt, movie_link AS ml, movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND lt.link LIKE '%follows%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id;