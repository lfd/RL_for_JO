SELECT * FROM movie_keyword AS mk, title AS t, kind_type AS kt, movie_companies AS mc WHERE kt.kind IN ('movie', 'episode') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;