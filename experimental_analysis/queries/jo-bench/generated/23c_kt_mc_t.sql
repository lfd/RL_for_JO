SELECT * FROM title AS t, kind_type AS kt, movie_companies AS mc WHERE kt.kind IN ('movie', 'tv movie', 'video movie', 'video game') AND t.production_year > 1990 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mc.movie_id AND mc.movie_id = t.id;