SELECT * FROM keyword AS k, movie_keyword AS mk, complete_cast AS cc WHERE k.keyword = 'computer-animation' AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;