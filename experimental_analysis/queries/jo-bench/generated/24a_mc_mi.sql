SELECT * FROM movie_companies AS mc, movie_info AS mi WHERE mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;