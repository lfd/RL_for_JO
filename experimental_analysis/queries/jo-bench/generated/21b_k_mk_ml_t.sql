SELECT * FROM movie_link AS ml, keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword = 'sequel' AND t.production_year BETWEEN 2000 AND 2010 AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;