SELECT * FROM comp_cast_type AS cct1, keyword AS k, cast_info AS ci, movie_keyword AS mk, complete_cast AS cc WHERE cct1.kind = 'cast' AND k.keyword IN ('superhero', 'sequel', 'second-part', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence') AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;