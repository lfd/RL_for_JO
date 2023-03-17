SELECT * FROM movie_info AS mi, title AS t, movie_companies AS mc WHERE mi.info LIKE 'USA:%200%' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;