SELECT * FROM title AS t, movie_keyword AS mk WHERE t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id;