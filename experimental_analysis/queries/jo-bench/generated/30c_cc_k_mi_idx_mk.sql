SELECT * FROM keyword AS k, movie_keyword AS mk, complete_cast AS cc, movie_info_idx AS mi_idx WHERE k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;