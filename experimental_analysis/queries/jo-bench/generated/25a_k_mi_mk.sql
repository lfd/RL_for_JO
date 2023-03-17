SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info AS mi WHERE k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND mi.info = 'Horror' AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;