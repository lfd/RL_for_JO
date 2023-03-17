SELECT * FROM keyword AS k, complete_cast AS cc, movie_keyword AS mk, movie_info AS mi WHERE k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;