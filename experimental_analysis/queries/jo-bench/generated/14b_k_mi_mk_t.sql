SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, movie_info AS mi WHERE k.keyword IN ('murder', 'murder-in-title') AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2010 AND (t.title LIKE '%murder%' OR t.title LIKE '%Murder%' OR t.title LIKE '%Mord%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;