SELECT * FROM link_type AS lt, keyword AS k, movie_keyword AS mk, movie_companies AS mc, title AS t, movie_link AS ml, company_type AS ct WHERE ct.kind != 'production companies' AND ct.kind IS NOT NULL AND k.keyword IN ('sequel', 'revenge', 'based-on-novel') AND mc.note IS NOT NULL AND t.production_year > 1950 AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;