SELECT * FROM movie_companies AS mc, movie_keyword AS mk, movie_info_idx AS mi_idx WHERE mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;