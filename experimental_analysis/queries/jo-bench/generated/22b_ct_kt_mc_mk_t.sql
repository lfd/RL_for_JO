SELECT * FROM movie_keyword AS mk, title AS t, kind_type AS kt, movie_companies AS mc, company_type AS ct WHERE kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2009 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;