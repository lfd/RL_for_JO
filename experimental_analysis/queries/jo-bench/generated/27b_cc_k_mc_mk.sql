SELECT * FROM keyword AS k, movie_keyword AS mk, movie_companies AS mc, complete_cast AS cc WHERE k.keyword = 'sequel' AND mc.note IS NULL AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;