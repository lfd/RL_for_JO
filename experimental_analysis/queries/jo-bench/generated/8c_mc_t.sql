SELECT * FROM movie_companies AS mc, title AS t WHERE t.id = mc.movie_id AND mc.movie_id = t.id;