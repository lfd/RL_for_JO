SELECT * FROM keyword AS k, movie_companies AS mc, movie_keyword AS mk WHERE k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;