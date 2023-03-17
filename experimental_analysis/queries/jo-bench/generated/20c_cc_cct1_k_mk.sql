SELECT * FROM comp_cast_type AS cct1, keyword AS k, movie_keyword AS mk, complete_cast AS cc WHERE cct1.kind = 'cast' AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence', 'magnet', 'web', 'claw', 'laser') AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;