SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t WHERE k.keyword = 'character-name-in-title' AND t.episode_nr >= 50 AND t.episode_nr < 100 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id;