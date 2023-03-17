SELECT * FROM cast_info AS ci, complete_cast AS cc, movie_keyword AS mk, aka_name AS an WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;