SELECT * FROM movie_keyword AS mk, keyword AS k WHERE k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND k.id = mk.keyword_id AND mk.keyword_id = k.id;