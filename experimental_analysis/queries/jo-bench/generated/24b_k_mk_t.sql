SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat', 'computer-animated-movie') AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = mk.movie_id AND mk.movie_id = t.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;