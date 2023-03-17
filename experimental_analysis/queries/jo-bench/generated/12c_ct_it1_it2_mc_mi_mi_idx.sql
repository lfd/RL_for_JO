SELECT * FROM company_type AS ct, info_type AS it2, movie_info_idx AS mi_idx, movie_info AS mi, info_type AS it1, movie_companies AS mc WHERE ct.kind = 'production companies' AND it1.info = 'genres' AND it2.info = 'rating' AND mi.info IN ('Drama', 'Horror', 'Western', 'Family') AND mi_idx.info > '7.0' AND mi.info_type_id = it1.id AND it1.id = mi.info_type_id AND mi_idx.info_type_id = it2.id AND it2.id = mi_idx.info_type_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;