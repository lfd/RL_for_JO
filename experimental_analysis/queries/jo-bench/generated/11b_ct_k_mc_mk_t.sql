SELECT * FROM company_type AS ct, movie_companies AS mc, title AS t, keyword AS k, movie_keyword AS mk WHERE ct.kind = 'production companies' AND k.keyword = 'sequel' AND mc.note IS NULL AND t.production_year = 1998 AND t.title LIKE '%Money%' AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;