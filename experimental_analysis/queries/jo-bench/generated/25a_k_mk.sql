SELECT * FROM keyword AS k, movie_keyword AS mk WHERE k.keyword IN ('murder', 'blood', 'gore', 'death', 'female-nudity') AND k.id = mk.keyword_id AND mk.keyword_id = k.id;