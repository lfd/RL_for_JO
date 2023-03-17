SELECT * FROM movie_keyword AS mk, cast_info AS ci, title AS t WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;