SELECT * FROM movie_info AS mi, title AS t, movie_keyword AS mk WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2010 AND (t.title LIKE '%murder%' OR t.title LIKE '%Murder%' OR t.title LIKE '%Mord%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id;