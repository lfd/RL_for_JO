SELECT * FROM comp_cast_type AS cct2, movie_info AS mi, keyword AS k, movie_keyword AS mk, complete_cast AS cc WHERE cct2.kind != 'complete+verified' AND k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;