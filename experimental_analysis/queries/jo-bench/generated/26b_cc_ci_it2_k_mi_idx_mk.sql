SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci, movie_info_idx AS mi_idx, info_type AS it2, complete_cast AS cc WHERE it2.info = 'rating' AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND mi_idx.info > '8.0' AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;