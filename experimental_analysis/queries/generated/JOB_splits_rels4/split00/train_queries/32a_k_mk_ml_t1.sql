SELECT * FROM movie_keyword AS mk, keyword AS k, title AS t1, movie_link AS ml WHERE k.keyword = '10,000-mile-club' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t1.id = mk.movie_id AND mk.movie_id = t1.id AND ml.movie_id = t1.id AND t1.id = ml.movie_id;