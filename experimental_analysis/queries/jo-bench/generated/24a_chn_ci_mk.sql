SELECT * FROM char_name AS chn, movie_keyword AS mk, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;