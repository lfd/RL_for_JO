SELECT * FROM keyword AS k, complete_cast AS cc, comp_cast_type AS cct1, movie_keyword AS mk WHERE cct1.kind = 'complete+verified' AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;