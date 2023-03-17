SELECT * FROM movie_keyword AS mk, keyword AS k, movie_companies AS mc, company_type AS ct, title AS t WHERE k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;