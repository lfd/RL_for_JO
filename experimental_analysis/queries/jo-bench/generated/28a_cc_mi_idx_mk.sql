SELECT * FROM complete_cast AS cc, movie_keyword AS mk, movie_info_idx AS mi_idx WHERE mi_idx.info < '8.5' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id;