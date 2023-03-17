SELECT * FROM cast_info AS ci, movie_info_idx AS mi_idx, movie_keyword AS mk, keyword AS k WHERE k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND mi_idx.info > '8.0' AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;