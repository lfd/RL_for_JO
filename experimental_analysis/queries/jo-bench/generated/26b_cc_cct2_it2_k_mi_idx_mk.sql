SELECT * FROM complete_cast AS cc, keyword AS k, movie_keyword AS mk, movie_info_idx AS mi_idx, info_type AS it2, comp_cast_type AS cct2 WHERE cct2.kind LIKE '%complete%' AND it2.info = 'rating' AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND mi_idx.info > '8.0' AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;