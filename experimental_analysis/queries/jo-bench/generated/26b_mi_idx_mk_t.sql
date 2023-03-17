SELECT * FROM movie_keyword AS mk, title AS t, movie_info_idx AS mi_idx WHERE mi_idx.info > '8.0' AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id;