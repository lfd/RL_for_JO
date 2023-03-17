SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci, name AS n, aka_name AS an, title AS t WHERE k.keyword = 'character-name-in-title' AND t.episode_nr >= 5 AND t.episode_nr < 100 AND an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;