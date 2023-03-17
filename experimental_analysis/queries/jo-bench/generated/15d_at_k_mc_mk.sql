SELECT * FROM movie_companies AS mc, aka_title AS at, keyword AS k, movie_keyword AS mk WHERE mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = at.movie_id AND at.movie_id = mk.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;