SELECT * FROM cast_info AS ci, title AS t WHERE ci.note IN ('(producer)', '(executive producer)') AND t.id = ci.movie_id AND ci.movie_id = t.id;