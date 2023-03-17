SELECT * FROM info_type AS it2, keyword AS k, movie_info_idx AS mi_idx, movie_keyword AS mk, complete_cast AS cc, title AS t WHERE it2.info = 'rating' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi_idx.info > '6.5' AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;