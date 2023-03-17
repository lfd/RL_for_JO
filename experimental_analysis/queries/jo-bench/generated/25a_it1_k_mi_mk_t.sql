SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, movie_info AS mi, info_type AS it1 WHERE it1.info = 'genres' AND k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND mi.info = 'Horror' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;