SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword IN ('sequel', 'revenge', 'based-on-novel') AND t.production_year > 1950 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id;