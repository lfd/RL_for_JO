SELECT * FROM movie_companies AS mc, cast_info AS ci, movie_keyword AS mk WHERE ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;