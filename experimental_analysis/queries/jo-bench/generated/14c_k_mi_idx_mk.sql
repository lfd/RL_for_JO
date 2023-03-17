SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info_idx AS mi_idx WHERE k.keyword IS NOT NULL AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi_idx.info < '8.5' AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;