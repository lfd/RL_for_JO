SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword = 'marvel-cinematic-universe' AND t.production_year > 2000 AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND t.id = mk.movie_id AND mk.movie_id = t.id;