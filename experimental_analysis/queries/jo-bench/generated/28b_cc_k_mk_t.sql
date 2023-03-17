SELECT * FROM keyword AS k, movie_keyword AS mk, complete_cast AS cc, title AS t WHERE k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;