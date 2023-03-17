SELECT * FROM movie_keyword AS mk, movie_info AS mi WHERE mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id;