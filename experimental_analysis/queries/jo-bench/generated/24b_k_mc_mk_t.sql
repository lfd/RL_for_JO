SELECT * FROM keyword AS k, movie_keyword AS mk, movie_companies AS mc, title AS t WHERE k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat', 'computer-animated-movie') AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;