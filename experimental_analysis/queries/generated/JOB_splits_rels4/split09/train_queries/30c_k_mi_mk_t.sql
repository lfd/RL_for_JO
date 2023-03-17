SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, movie_info AS mi WHERE k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info IN ('Horror', 'Action', 'Sci-Fi', 'Thriller', 'Crime', 'War') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;