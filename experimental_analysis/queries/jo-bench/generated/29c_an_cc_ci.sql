SELECT * FROM cast_info AS ci, aka_name AS an, complete_cast AS cc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;