SELECT * FROM movie_keyword AS mk, title AS t, movie_info AS mi WHERE mi.info IN ('Germany', 'German', 'USA', 'American') AND t.production_year > 2008 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;