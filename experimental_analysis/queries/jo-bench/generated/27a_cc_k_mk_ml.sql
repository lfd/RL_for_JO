SELECT * FROM keyword AS k, movie_keyword AS mk, complete_cast AS cc, movie_link AS ml WHERE k.keyword = 'sequel' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id;