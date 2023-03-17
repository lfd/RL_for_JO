SELECT * FROM kind_type AS kt, movie_info AS mi, title AS t, info_type AS it1 WHERE it1.info = 'countries' AND kt.kind = 'movie' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2010 AND (t.title LIKE '%murder%' OR t.title LIKE '%Murder%' OR t.title LIKE '%Mord%') AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;