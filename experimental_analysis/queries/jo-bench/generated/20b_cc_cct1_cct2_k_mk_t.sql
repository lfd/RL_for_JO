SELECT * FROM comp_cast_type AS cct2, comp_cast_type AS cct1, keyword AS k, movie_keyword AS mk, complete_cast AS cc, title AS t WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND k.keyword IN ('superhero', 'sequel', 'second-part', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence') AND t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;