SELECT * FROM movie_companies AS mc, movie_info AS mi WHERE mc.note IS NULL AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'English') AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;