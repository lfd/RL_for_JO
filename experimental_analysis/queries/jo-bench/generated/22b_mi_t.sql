SELECT * FROM movie_info AS mi, title AS t WHERE mi.info IN ('Germany', 'German', 'USA', 'American') AND t.production_year > 2009 AND t.id = mi.movie_id AND mi.movie_id = t.id;