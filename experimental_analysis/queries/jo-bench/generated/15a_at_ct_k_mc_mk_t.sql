SELECT * FROM company_type AS ct, movie_companies AS mc, aka_title AS at, keyword AS k, movie_keyword AS mk, title AS t WHERE mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND t.production_year > 2000 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;