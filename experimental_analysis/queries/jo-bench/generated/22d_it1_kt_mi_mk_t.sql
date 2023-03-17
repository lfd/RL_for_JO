SELECT * FROM movie_keyword AS mk, title AS t, kind_type AS kt, movie_info AS mi, info_type AS it1 WHERE it1.info = 'countries' AND kt.kind IN ('movie', 'episode') AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;