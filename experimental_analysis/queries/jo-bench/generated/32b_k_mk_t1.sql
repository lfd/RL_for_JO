SELECT * FROM movie_keyword AS mk, keyword AS k, title AS t1 WHERE k.keyword = 'character-name-in-title' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t1.id = mk.movie_id AND mk.movie_id = t1.id;