SELECT * FROM movie_info AS mi, kind_type AS kt, title AS t WHERE kt.kind IN ('movie', 'episode') AND mi.info IN ('Germany', 'German', 'USA', 'American') AND t.production_year > 2008 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id;