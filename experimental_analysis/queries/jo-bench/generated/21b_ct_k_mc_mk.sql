SELECT * FROM movie_companies AS mc, keyword AS k, movie_keyword AS mk, company_type AS ct WHERE ct.kind = 'production companies' AND k.keyword = 'sequel' AND mc.note IS NULL AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;