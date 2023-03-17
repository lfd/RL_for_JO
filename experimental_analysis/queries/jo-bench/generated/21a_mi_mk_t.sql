SELECT * FROM movie_keyword AS mk, movie_info AS mi, title AS t WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German') AND t.production_year BETWEEN 1950 AND 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = t.id AND t.id = mi.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;