SELECT * FROM movie_info AS mi, movie_companies AS mc WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;