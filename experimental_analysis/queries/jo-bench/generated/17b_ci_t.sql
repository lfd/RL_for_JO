SELECT * FROM cast_info AS ci, title AS t WHERE ci.movie_id = t.id AND t.id = ci.movie_id;