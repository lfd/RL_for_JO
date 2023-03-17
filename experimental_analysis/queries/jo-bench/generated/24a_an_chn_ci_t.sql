SELECT * FROM aka_name AS an, cast_info AS ci, title AS t, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND t.production_year > 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;