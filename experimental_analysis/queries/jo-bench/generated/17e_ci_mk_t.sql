SELECT * FROM cast_info AS ci, movie_keyword AS mk, title AS t WHERE ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;