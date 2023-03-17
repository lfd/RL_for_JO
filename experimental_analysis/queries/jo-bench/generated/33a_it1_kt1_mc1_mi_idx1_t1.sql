SELECT * FROM info_type AS it1, kind_type AS kt1, movie_info_idx AS mi_idx1, title AS t1, movie_companies AS mc1 WHERE it1.info = 'rating' AND kt1.kind IN ('tv series') AND it1.id = mi_idx1.info_type_id AND mi_idx1.info_type_id = it1.id AND t1.id = mi_idx1.movie_id AND mi_idx1.movie_id = t1.id AND kt1.id = t1.kind_id AND t1.kind_id = kt1.id AND t1.id = mc1.movie_id AND mc1.movie_id = t1.id AND mi_idx1.movie_id = mc1.movie_id AND mc1.movie_id = mi_idx1.movie_id;