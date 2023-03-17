SELECT * FROM movie_info AS mi, movie_companies AS mc, title AS t WHERE mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;