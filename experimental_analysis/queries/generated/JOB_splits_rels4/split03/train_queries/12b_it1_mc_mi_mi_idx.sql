SELECT * FROM info_type AS it1, movie_info_idx AS mi_idx, movie_info AS mi, movie_companies AS mc WHERE it1.info = 'budget' AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;