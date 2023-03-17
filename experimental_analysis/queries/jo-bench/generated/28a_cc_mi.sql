SELECT * FROM movie_info AS mi, complete_cast AS cc WHERE mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Danish', 'Norwegian', 'German', 'USA', 'American') AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;