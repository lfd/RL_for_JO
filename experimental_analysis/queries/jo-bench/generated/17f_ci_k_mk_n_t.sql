SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, cast_info AS ci, name AS n WHERE k.keyword = 'character-name-in-title' AND n.name LIKE '%B%' AND n.id = ci.person_id AND ci.person_id = n.id AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;