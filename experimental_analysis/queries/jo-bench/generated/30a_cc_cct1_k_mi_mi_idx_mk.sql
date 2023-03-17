SELECT * FROM keyword AS k, movie_keyword AS mk, complete_cast AS cc, movie_info_idx AS mi_idx, comp_cast_type AS cct1, movie_info AS mi WHERE cct1.kind IN ('cast', 'crew') AND k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mi_idx.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND cc.movie_id = mi_idx.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;