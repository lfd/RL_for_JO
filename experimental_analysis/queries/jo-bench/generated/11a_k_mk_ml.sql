SELECT * FROM movie_keyword AS mk, movie_link AS ml, keyword AS k WHERE k.keyword = 'sequel' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id;