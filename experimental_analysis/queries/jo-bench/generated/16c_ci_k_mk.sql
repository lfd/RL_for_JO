SELECT * FROM movie_keyword AS mk, cast_info AS ci, keyword AS k WHERE k.keyword = 'character-name-in-title' AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id;