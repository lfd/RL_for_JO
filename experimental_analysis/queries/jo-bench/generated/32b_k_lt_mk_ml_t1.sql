SELECT * FROM movie_keyword AS mk, title AS t1, movie_link AS ml, keyword AS k, link_type AS lt WHERE k.keyword = 'character-name-in-title' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t1.id = mk.movie_id AND mk.movie_id = t1.id AND ml.movie_id = t1.id AND t1.id = ml.movie_id AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id;