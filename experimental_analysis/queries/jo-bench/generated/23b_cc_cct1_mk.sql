SELECT * FROM movie_keyword AS mk, comp_cast_type AS cct1, complete_cast AS cc WHERE cct1.kind = 'complete+verified' AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;