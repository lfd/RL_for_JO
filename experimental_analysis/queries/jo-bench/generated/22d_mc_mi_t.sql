SELECT * FROM movie_info AS mi, movie_companies AS mc, title AS t WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;