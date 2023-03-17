SELECT * FROM kind_type AS kt, keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword IS NOT NULL AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND kt.kind IN ('movie', 'episode') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;