SELECT * FROM kind_type AS kt, title AS t, movie_companies AS mc WHERE kt.kind = 'movie' AND kt.id = t.kind_id AND t.kind_id = kt.id AND mc.movie_id = t.id AND t.id = mc.movie_id;