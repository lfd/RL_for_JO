SELECT * FROM movie_companies AS mc, movie_info_idx AS mi_idx WHERE mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%' AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id;