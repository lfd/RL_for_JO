SELECT * FROM title AS t, keyword AS k, movie_keyword AS mk WHERE k.keyword = 'computer-animation' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mk.movie_id AND mk.movie_id = t.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;