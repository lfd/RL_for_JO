SELECT * FROM movie_info AS mi, title AS t, movie_info_idx AS mi_idx WHERE mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id;