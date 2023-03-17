SELECT * FROM movie_companies AS mc, aka_title AS at, title AS t WHERE mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND t.production_year BETWEEN 2005 AND 2010 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id;