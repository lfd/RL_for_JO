SELECT * FROM info_type AS it1, movie_info AS mi, movie_info_idx AS mi_idx, movie_companies AS mc WHERE it1.info = 'countries' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND mi_idx.info < '7.0' AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;