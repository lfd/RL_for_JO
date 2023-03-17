SELECT * FROM kind_type AS kt, movie_info_idx AS mi_idx, title AS t WHERE kt.kind = 'movie' AND mi_idx.info > '6.0' AND t.production_year > 2010 AND (t.title LIKE '%murder%' OR t.title LIKE '%Murder%' OR t.title LIKE '%Mord%') AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id;