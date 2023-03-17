SELECT * FROM keyword AS k, movie_keyword AS mk, movie_companies AS mc, cast_info AS ci, aka_name AS an, name AS n WHERE k.keyword = 'character-name-in-title' AND an.person_id = n.id AND n.id = an.person_id AND n.id = ci.person_id AND ci.person_id = n.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id;