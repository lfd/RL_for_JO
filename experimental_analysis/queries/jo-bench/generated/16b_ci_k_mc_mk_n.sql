SELECT * FROM name AS n, keyword AS k, movie_keyword AS mk, cast_info AS ci, movie_companies AS mc WHERE k.keyword = 'character-name-in-title' AND n.id = ci.person_id AND ci.person_id = n.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;