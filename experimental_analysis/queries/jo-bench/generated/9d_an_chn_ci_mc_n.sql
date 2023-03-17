SELECT * FROM cast_info AS ci, aka_name AS an, name AS n, char_name AS chn, movie_companies AS mc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;