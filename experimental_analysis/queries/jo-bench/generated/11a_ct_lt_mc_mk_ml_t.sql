SELECT * FROM movie_keyword AS mk, title AS t, movie_link AS ml, link_type AS lt, movie_companies AS mc, company_type AS ct WHERE ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2000 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;