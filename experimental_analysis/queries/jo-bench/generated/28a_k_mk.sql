SELECT * FROM keyword AS k, movie_keyword AS mk WHERE k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND k.id = mk.keyword_id AND mk.keyword_id = k.id;