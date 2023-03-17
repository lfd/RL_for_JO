SELECT * FROM movie_info_idx AS mi_idx, title AS t, movie_keyword AS mk, keyword AS k WHERE k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi_idx.info < '7.0' AND t.production_year > 2009 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;