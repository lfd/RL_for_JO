SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info_idx AS mi_idx WHERE k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence', 'magnet', 'web', 'claw', 'laser') AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;