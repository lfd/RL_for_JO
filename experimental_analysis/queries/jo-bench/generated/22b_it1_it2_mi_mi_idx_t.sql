SELECT * FROM movie_info AS mi, info_type AS it1, movie_info_idx AS mi_idx, info_type AS it2, title AS t WHERE it1.info = 'countries' AND it2.info = 'rating' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND mi_idx.info < '7.0' AND t.production_year > 2009 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;