SELECT * FROM movie_keyword AS mk, keyword AS k WHERE k.id = mk.keyword_id AND mk.keyword_id = k.id;