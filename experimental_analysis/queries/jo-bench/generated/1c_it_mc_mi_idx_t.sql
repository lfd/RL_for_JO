SELECT * FROM info_type AS it, movie_info_idx AS mi_idx, movie_companies AS mc, title AS t WHERE it.info = 'top 250 rank' AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%' AND mc.note LIKE '%(co-production)%' AND t.production_year > 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mc.movie_id AND it.id = mi_idx.info_type_id AND mi_idx.info_type_id = it.id;