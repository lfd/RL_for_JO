SELECT * FROM complete_cast AS cc, movie_companies AS mc WHERE mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;