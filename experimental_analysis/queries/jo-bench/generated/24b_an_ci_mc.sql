SELECT * FROM movie_companies AS mc, aka_name AS an, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;