SELECT * FROM keyword AS k, kind_type AS kt, movie_keyword AS mk, title AS t WHERE k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND kt.kind = 'movie' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;