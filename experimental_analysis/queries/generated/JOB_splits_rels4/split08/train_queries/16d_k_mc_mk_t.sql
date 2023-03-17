SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, movie_companies AS mc WHERE k.keyword = 'character-name-in-title' AND t.episode_nr >= 5 AND t.episode_nr < 100 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;