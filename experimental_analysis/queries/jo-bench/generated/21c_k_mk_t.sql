SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword = 'sequel' AND t.production_year BETWEEN 1950 AND 2010 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id;