SELECT * FROM aka_title AS at, movie_keyword AS mk, keyword AS k WHERE mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;