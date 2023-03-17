SELECT * FROM movie_companies AS mc, keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword = 'sequel' AND mc.note IS NULL AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;