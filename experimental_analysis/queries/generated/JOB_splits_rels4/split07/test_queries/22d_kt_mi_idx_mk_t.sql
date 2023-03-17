SELECT * FROM movie_info_idx AS mi_idx, title AS t, movie_keyword AS mk, kind_type AS kt WHERE kt.kind IN ('movie', 'episode') AND mi_idx.info < '8.5' AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id;